use std::io::Cursor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioFormat {
    Wav,
    Mp3,
    Flac,
    OggVorbis,
    Unknown,
}

impl AudioFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
            Self::Flac => "flac",
            Self::OggVorbis => "ogg",
            Self::Unknown => "unknown",
        }
    }
}

pub fn detect_format(data: &[u8]) -> AudioFormat {
    if data.is_empty() {
        return AudioFormat::Unknown;
    }

    // Formats with 4-byte magic sequences
    if data.len() >= 4 {
        if &data[0..4] == b"RIFF" && data.len() >= 12 && &data[8..12] == b"WAVE" {
            return AudioFormat::Wav;
        }
        if &data[0..4] == b"fLaC" {
            return AudioFormat::Flac;
        }
        if &data[0..4] == b"OggS" {
            return AudioFormat::OggVorbis;
        }
    }

    // MP3: ID3v2 tag header (3 bytes) or sync word (2 bytes)
    if data.len() >= 3 && &data[0..3] == b"ID3" {
        return AudioFormat::Mp3;
    }
    if data.len() >= 2 {
        let sync = u16::from_be_bytes([data[0], data[1]]);
        if sync == 0xFFFB || sync == 0xFFFA || sync == 0xFFF3 || sync == 0xFFF2 {
            return AudioFormat::Mp3;
        }
    }

    AudioFormat::Unknown
}

pub fn linear_resample(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz {
        return samples.to_vec();
    }
    if samples.is_empty() || from_hz == 0 || to_hz == 0 {
        return vec![];
    }

    let ratio = to_hz as f64 / from_hz as f64;
    let new_len = ((samples.len() as f64) * ratio).round() as usize;
    if new_len == 0 {
        return vec![];
    }

    let mut output = Vec::with_capacity(new_len);
    for i in 0..new_len {
        let src_idx = (i as f64 / ratio).floor() as usize;
        let src_idx = src_idx.min(samples.len() - 1);
        output.push(samples[src_idx]);
    }

    output
}

/// High-quality resampling using rubato's synchronous FFT resampler.
/// Produces significantly better audio quality than `linear_resample`,
/// especially for inputs with significant high-frequency content.
pub fn rubato_resample(samples: &[f32], from_hz: u32, to_hz: u32) -> Result<Vec<f32>, String> {
    use rubato::{Fft, FixedSync, Resampler};
    use rubato::audioadapter_buffers::direct::InterleavedSlice;

    if from_hz == to_hz {
        return Ok(samples.to_vec());
    }
    if samples.is_empty() || from_hz == 0 || to_hz == 0 {
        return Ok(vec![]);
    }

    let channels = 1;
    let chunk_size = 1024;

    let mut resampler = Fft::<f32>::new(
        to_hz as usize,
        from_hz as usize,
        chunk_size,
        channels,
        2,                     // max_frames_res_ratio
        FixedSync::Input,
    )
    .map_err(|e| format!("rubato Fft::new: {e}"))?;

    let nbr_input_frames = samples.len();
    let input_adapter =
        InterleavedSlice::new(samples, channels, nbr_input_frames)
            .map_err(|e| format!("rubato input adapter: {e}"))?;

    let output_len = resampler.process_all_needed_output_len(nbr_input_frames);
    let mut output_buf = vec![0.0f32; output_len * channels];
    let mut output_adapter =
        InterleavedSlice::new_mut(&mut output_buf, channels, output_len)
            .map_err(|e| format!("rubato output adapter: {e}"))?;

    let (_frames_read, frames_written) = resampler
        .process_all_into_buffer(&input_adapter, &mut output_adapter, nbr_input_frames, None)
        .map_err(|e| format!("rubato process_all: {e}"))?;

    output_buf.truncate(frames_written * channels);
    Ok(output_buf)
}

pub fn decode_to_pcm16k_mono(data: &[u8]) -> Result<(Vec<f32>, u32), String> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let format = detect_format(data);

    match format {
        AudioFormat::Unknown => Err("unknown audio format".to_string()),
        AudioFormat::Wav => decode_wav(data),
        _ => {
            // Ownership transfer: symphonia's MediaSourceStream requires 'static data
            let cursor = Cursor::new(data.to_vec());
            let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

            let mut hint = Hint::new();
            match format {
                AudioFormat::Mp3 => { hint.with_extension("mp3"); }
                AudioFormat::Flac => { hint.with_extension("flac"); }
                AudioFormat::OggVorbis => { hint.with_extension("ogg"); }
                _ => {}
            };

            let format_opts = FormatOptions::default();
            let metadata_opts = MetadataOptions::default();
            let decoder_opts = DecoderOptions::default();

            let probed = symphonia::default::get_probe()
                .format(&hint, mss, &format_opts, &metadata_opts)
                .map_err(|e| format!("symphonia probe: {}", e))?;

            let mut format = probed.format;

            let track = format
                .tracks()
                .iter()
                .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
                .ok_or("no audio track found")?;

            let track_id = track.id;
            let codec_params = track.codec_params.clone();

            let sample_rate = codec_params.sample_rate.unwrap_or(16000);
            let channels = codec_params.channels.map(|c| c.count()).unwrap_or(1);

            let mut decoder = symphonia::default::get_codecs()
                .make(&codec_params, &decoder_opts)
                .map_err(|e| format!("symphonia decoder: {}", e))?;

            let mut all_samples: Vec<f32> = Vec::new();

            loop {
                let packet = match format.next_packet() {
                    Ok(p) => p,
                    Err(symphonia::core::errors::Error::IoError(ref e))
                        if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                    {
                        break;
                    }
                    Err(_) => break,
                };

                if packet.track_id() != track_id {
                    continue;
                }

                let decoded = match decoder.decode(&packet) {
                    Ok(d) => d,
                    Err(_) => continue,
                };

                let spec = *decoded.spec();
                let duration = decoded.capacity() as u64;

                let mut sample_buf = SampleBuffer::<f32>::new(duration, spec);
                sample_buf.copy_interleaved_ref(decoded);

                let samples = sample_buf.samples();
                if channels == 2 {
                    for i in (0..samples.len()).step_by(2) {
                        let mono = (samples[i] + samples[i + 1]) / 2.0;
                        all_samples.push(mono);
                    }
                } else {
                    all_samples.extend_from_slice(samples);
                }
            }

            let mono_16k = if sample_rate != 16000 {
                rubato_resample(&all_samples, sample_rate, 16000)?
            } else {
                all_samples
            };

            Ok((mono_16k, 16000))
        }
    }
}

fn decode_wav(data: &[u8]) -> Result<(Vec<f32>, u32), String> {
    if data.len() < 44 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err("not a valid WAV file".to_string());
    }
    let mut pos = 12;
    let mut sample_rate = 0u32;
    let mut num_channels = 0u16;
    let mut bits_per_sample = 0u16;
    let mut audio_data_pos = 0;
    let mut audio_data_len = 0;
    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size =
            u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]);
        if chunk_id == b"fmt " {
            num_channels = u16::from_le_bytes([data[pos + 10], data[pos + 11]]);
            sample_rate =
                u32::from_le_bytes([data[pos + 12], data[pos + 13], data[pos + 14], data[pos + 15]]);
            bits_per_sample = u16::from_le_bytes([data[pos + 22], data[pos + 23]]);
        } else if chunk_id == b"data" {
            audio_data_pos = pos + 8;
            audio_data_len = chunk_size as usize;
            break;
        }
        pos += 8 + chunk_size as usize;
    }
    if audio_data_pos == 0 {
        return Err("no audio data chunk found".to_string());
    }
    if bits_per_sample != 16 {
        return Err(format!(
            "unsupported bits per sample {} (only 16-bit supported)",
            bits_per_sample
        ));
    }

    let mut samples = Vec::with_capacity(audio_data_len / 2);
    for i in (0..audio_data_len).step_by(2) {
        let s = i16::from_le_bytes([
            data[audio_data_pos + i],
            data[audio_data_pos + i + 1],
        ]);
        samples.push(s as f32 / 32768.0);
    }

    if num_channels == 2 {
        let mut mono = Vec::with_capacity(samples.len() / 2);
        for i in (0..samples.len()).step_by(2) {
            mono.push((samples[i] + samples[i + 1]) / 2.0);
        }
        samples = mono;
    }

    let mono_16k = if sample_rate != 16000 {
        linear_resample(&samples, sample_rate, 16000)
    } else {
        samples
    };

    Ok((mono_16k, sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_format_wav() {
        let data = b"RIFF\x00\x00\x00\x00WAVE".to_vec();
        assert_eq!(detect_format(&data), AudioFormat::Wav);
    }

    #[test]
    fn test_detect_format_flac() {
        let data = b"fLaC".to_vec();
        assert_eq!(detect_format(&data), AudioFormat::Flac);
    }

    #[test]
    fn test_detect_format_ogg() {
        let data = b"OggS".to_vec();
        assert_eq!(detect_format(&data), AudioFormat::OggVorbis);
    }

    #[test]
    fn test_detect_format_mp3_id3() {
        let data = b"ID3".to_vec();
        assert_eq!(detect_format(&data), AudioFormat::Mp3);
    }

    #[test]
    fn test_detect_format_mp3_sync() {
        let data = vec![0xFF, 0xFB, 0x90];
        assert_eq!(detect_format(&data), AudioFormat::Mp3);
    }

    #[test]
    fn test_detect_format_unknown() {
        let data = b"hello world".to_vec();
        assert_eq!(detect_format(&data), AudioFormat::Unknown);
    }

    #[test]
    fn test_linear_resample_preserves_length_ratio() {
        let input: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let output = linear_resample(&input, 10, 20);
        assert_eq!(output.len(), 200);
    }

    #[test]
    fn test_linear_resample_sine_wave_1hz_to_2hz() {
        let sample_rate = 10usize;
        let num_samples = 20;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (std::f32::consts::TAU * t).sin()
            })
            .collect();

        let output = linear_resample(&input, 10, 20);

        assert!(output.len() > 0);
    }
}