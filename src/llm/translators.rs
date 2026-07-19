use crate::canonical::{
    CanonicalLlmRequest, CanonicalLlmResponse, ContentBlock,
    MessageRole, StopReason, TextBlock, UsageInfo,
};
use crate::llm::{ChatMessage, ChatRequest, ChatResponse};

pub fn canonical_to_chat_request(req: &CanonicalLlmRequest) -> Result<ChatRequest, String> {
    let mut messages = Vec::new();
    for sb in &req.system {
        messages.push(ChatMessage { role: "system".into(), content: sb.text.clone() });
    }
    for m in &req.messages {
        let role = match m.role {
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
            MessageRole::System => "system",
        };
        let content = blocks_to_text(&m.content);
        messages.push(ChatMessage { role: role.into(), content });
    }
    Ok(ChatRequest {
        model: req.model.clone(),
        messages,
        temperature: req.sampling.temperature,
        max_tokens: req.sampling.max_tokens,
        top_p: req.sampling.top_p,
        top_k: req.sampling.top_k,
        stream: req.stream,
        stop: if req.stop_sequences.is_empty() { None } else { Some(req.stop_sequences.clone()) },
    })
}

fn blocks_to_text(blocks: &[ContentBlock]) -> String {
    let mut out = String::new();
    for b in blocks {
        match b {
            ContentBlock::Text(t) => out.push_str(&t.text),
            ContentBlock::Reasoning(r) => out.push_str(&r.text),
            other => { if let Ok(j) = serde_json::to_string(other) { out.push_str(&j); } }
        }
    }
    out
}

pub fn chat_response_to_canonical(model_id: &str, resp: ChatResponse) -> CanonicalLlmResponse {
    let first_choice = resp.choices.into_iter().next();
    let content = first_choice.as_ref().map(|c| {
        ContentBlock::Text(TextBlock { text: c.message.content.clone() })
    }).unwrap_or_else(|| ContentBlock::Text(TextBlock { text: String::new() }));

    let stop_reason = match first_choice.and_then(|c| c.finish_reason) {
        Some(ref s) if s == "stop" => StopReason::EndTurn,
        Some(ref s) if s == "length" => StopReason::MaxTokens,
        Some(ref s) if s == "tool_calls" => StopReason::ToolUse,
        Some(other) => StopReason::Other(other),
        None => StopReason::EndTurn,
    };

    CanonicalLlmResponse {
        id: resp.id,
        model: model_id.to_string(),
        stop_reason,
        content: vec![content],
        usage: resp.usage.map(|u| UsageInfo {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            extensions: [("total_tokens".to_string(), u.total_tokens.into())]
                .into_iter().collect(),
            ..Default::default()
        }).unwrap_or_default(),
        extensions: Default::default(),
    }
}