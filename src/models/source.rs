use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::mpsc::UnboundedSender;

#[derive(Debug, Clone)]
pub struct ParsedReference {
    pub model_id: String,
    pub revision: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub bytes_downloaded: u64,
    pub total_bytes: Option<u64>,
    pub current_file: String,
}

#[async_trait]
pub trait Source: Send + Sync {
    fn id(&self) -> &str;
    fn name(&self) -> &str;
    /// Optional base URL / endpoint of the source. Used by the
    /// `Catalog` to surface the resolved endpoint on `/v1/models`
    /// entries, and by `HuggingFaceSource` to redirect downloads
    /// to a mirror via the `HF_ENDPOINT` env var. Sources that
    /// don't have a configurable endpoint return `None`.
    fn base_url(&self) -> Option<&str> {
        None
    }
    fn parse_reference(&self, reference: &str) -> Result<ParsedReference, String>;
    async fn download(
        &self,
        reference: &ParsedReference,
        dest_dir: &Path,
        progress_tx: UnboundedSender<DownloadProgress>,
    ) -> Result<(), String>;
}

pub struct SourceRegistry {
    sources: HashMap<String, Arc<dyn Source>>,
}

impl SourceRegistry {
    pub fn new() -> Self {
        Self { sources: HashMap::new() }
    }

    pub fn with_defaults() -> Self {
        Self::with_endpoints(&std::collections::HashMap::new())
    }

    /// Build a `SourceRegistry` with the default HuggingFace and ModelScope
    /// sources, optionally configured with per-source base URL maps.
    ///
    /// Three HF-family sources are registered so the UI can offer a
    /// China-friendly mirror out of the box:
    /// - `huggingface`  → https://huggingface.co (default)
    /// - `hf-mirror`    → https://hf-mirror.com (China-friendly CDN)
    /// - `modelscope`   → https://modelscope.cn (Chinese repo)
    pub fn with_endpoints(
        endpoints: &std::collections::HashMap<String, String>,
    ) -> Self {
        let mut r = Self::new();

        // Primary HuggingFace — `https://huggingface.co` by default,
        // override via `endpoints["huggingface"]`.
        let hf_endpoint = endpoints.get("huggingface").cloned();
        r.register(Arc::new(super::huggingface::HuggingFaceSource::with_endpoint(
            std::path::PathBuf::from("huggingface-cli"),
            hf_endpoint,
        )));

        // HF mirror at hf-mirror.com — same protocol as HF, just a
        // CDN-fronted endpoint. Falls back to the user-supplied
        // override if present.
        let mirror_endpoint = endpoints
            .get("hf-mirror")
            .cloned()
            .or_else(|| Some("https://hf-mirror.com".to_string()));
        r.register(Arc::new(super::huggingface::HuggingFaceSource::with_identity(
            std::path::PathBuf::from("huggingface-cli"),
            mirror_endpoint,
            "hf-mirror",
            "HF Mirror",
        )));

        // ModelScope — Chinese alternative.
        let ms_endpoint = endpoints.get("modelscope").cloned();
        r.register(Arc::new(super::modelscope::ModelScopeSource::with_endpoint(
            std::path::PathBuf::from("modelscope-cli"),
            ms_endpoint,
        )));
        r
    }

    pub fn register(&mut self, source: Arc<dyn Source>) {
        self.sources.insert(source.id().to_string(), source);
    }

    pub fn get(&self, id: &str) -> Option<Arc<dyn Source>> {
        self.sources.get(id).cloned()
    }

    pub fn list(&self) -> Vec<Arc<dyn Source>> {
        self.sources.values().cloned().collect()
    }

    /// Return `(id, base_url)` for every registered source, sorted
    /// by id. Sources without a `base_url` carry `None`. Used by
    /// the planned `/v1/catalog/sources` endpoint and by tests that
    /// need to assert endpoint wiring (e.g. `with_endpoints`
    /// passing a HuggingFace mirror to the HuggingFaceSource).
    pub fn list_endpoints(&self) -> Vec<(String, Option<String>)> {
        let mut out: Vec<(String, Option<String>)> = self
            .sources
            .iter()
            .map(|(id, src)| (id.clone(), src.base_url().map(String::from)))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    pub fn parse_reference(&self, ref_str: &str) -> Result<(Arc<dyn Source>, ParsedReference), String> {
        let (source_id, rest) = if let Some((src, rest)) = ref_str.split_once(':') {
            (src, rest)
        } else {
            // Default source: ModelScope (Alibaba's open model hub).
            // Users behind the GFW can override with a `hf-mirror:`
            // or `huggingface:` prefix.
            ("modelscope", ref_str)
        };

        let source = self
            .get(source_id)
            .ok_or_else(|| format!("unknown source: {source_id}"))?;

        let parsed = source.parse_reference(rest)?;
        Ok((source, parsed))
    }

    /// Resolve which source a pull request should use, given a
    /// `CatalogConfig`. Lookup order: per-model > per-engine >
    /// default. Returns an error if the resolved source id is not
    /// registered in this registry.
    pub fn resolve_for(
        &self,
        engine: crate::models::EngineType,
        model_id: &str,
        config: &crate::models::catalog::CatalogConfig,
    ) -> Result<Arc<dyn Source>, String> {
        let id = config.resolve_source_id(engine, model_id);
        self.get(&id)
            .ok_or_else(|| format!("source not registered: {id}"))
    }
}

impl Default for SourceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    struct DummySource {
        id: String,
    }

    #[async_trait]
    impl Source for DummySource {
        fn id(&self) -> &str { &self.id }
        fn name(&self) -> &str { &self.id }
        fn parse_reference(&self, r: &str) -> Result<ParsedReference, String> {
            Ok(ParsedReference { model_id: r.to_string(), revision: None })
        }
        async fn download(
            &self,
            _r: &ParsedReference,
            _d: &Path,
            _p: UnboundedSender<DownloadProgress>,
        ) -> Result<(), String> {
            Ok(())
        }
    }

    fn dummy(id: &str) -> Arc<dyn Source> {
        Arc::new(DummySource { id: id.to_string() })
    }

    #[test]
    fn test_source_registry_register_and_get() {
        let mut r = SourceRegistry::new();
        r.register(dummy("test"));
        assert!(r.get("test").is_some());
        assert!(r.get("nonexistent").is_none());
    }

    #[test]
    fn test_source_registry_list() {
        let mut r = SourceRegistry::new();
        r.register(dummy("a"));
        r.register(dummy("b"));
        let ids: Vec<String> = r.list().iter().map(|s| s.id().to_string()).collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"a".to_string()));
        assert!(ids.contains(&"b".to_string()));
    }

    #[test]
    fn test_source_registry_default_includes_huggingface() {
        let r = SourceRegistry::with_defaults();
        assert!(r.get("huggingface").is_some());
    }

    #[test]
    fn test_parse_reference_defaults_to_modelscope() {
        let r = SourceRegistry::new();
        let mut r = r;
        r.register(dummy("modelscope"));
        let (src, _parsed) = r.parse_reference("Qwen/X").unwrap();
        assert_eq!(src.id(), "modelscope");
    }

    #[test]
    fn test_parse_reference_with_source_prefix() {
        let mut r = SourceRegistry::new();
        r.register(dummy("huggingface"));
        r.register(dummy("modelscope"));
        let (src, _) = r.parse_reference("modelscope:Qwen/X").unwrap();
        assert_eq!(src.id(), "modelscope");
    }

    #[test]
    fn test_parse_reference_unknown_source_returns_error() {
        let r = SourceRegistry::new();
        let result = r.parse_reference("unknown:Qwen/X");
        assert!(result.is_err());
        assert!(result.err().unwrap().contains("unknown source"));
    }

    #[test]
    fn test_resolve_for_returns_engine_override() {
        use crate::models::catalog::CatalogConfig;
        use crate::models::EngineType;
        use std::collections::HashMap;

        let mut r = SourceRegistry::new();
        r.register(dummy("huggingface"));
        r.register(dummy("modelscope"));
        let mut per_engine = HashMap::new();
        per_engine.insert(EngineType::Llm, "modelscope".to_string());
        let cfg = CatalogConfig {
            default_source_id: "huggingface".to_string(),
            per_engine_source: per_engine,
            per_model_source: HashMap::new(),
        };
        let s = r.resolve_for(EngineType::Llm, "x", &cfg).unwrap();
        assert_eq!(s.id(), "modelscope");
    }

    #[test]
    fn test_resolve_for_returns_err_for_missing_source() {
        use crate::models::catalog::CatalogConfig;
        use crate::models::EngineType;
        use std::collections::HashMap;

        let r = SourceRegistry::new();
        let cfg = CatalogConfig {
            default_source_id: "nonexistent".to_string(),
            per_engine_source: HashMap::new(),
            per_model_source: HashMap::new(),
        };
        let err = match r.resolve_for(EngineType::Llm, "x", &cfg) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        assert!(err.contains("not registered"));
    }

    #[test]
    fn test_with_endpoints_passes_hf_endpoint_to_registered_source() {
        use std::collections::HashMap;
        let mut endpoints = HashMap::new();
        endpoints.insert("huggingface".to_string(), "https://hf-mirror.com".to_string());
        let r = SourceRegistry::with_endpoints(&endpoints);
        let endpoints_listed = r.list_endpoints();
        // Three sources: huggingface, hf-mirror (always has a default
        // URL), and modelscope.
        assert_eq!(endpoints_listed.len(), 3);
        // huggingface should carry the custom endpoint.
        let hf = endpoints_listed.iter().find(|(id, _)| id == "huggingface").unwrap();
        assert_eq!(hf.1.as_deref(), Some("https://hf-mirror.com"));
        // hf-mirror always has a default URL.
        let mirror = endpoints_listed.iter().find(|(id, _)| id == "hf-mirror").unwrap();
        assert_eq!(mirror.1.as_deref(), Some("https://hf-mirror.com"));
        // modelscope has no custom endpoint → None.
        let ms = endpoints_listed.iter().find(|(id, _)| id == "modelscope").unwrap();
        assert_eq!(ms.1, None);
    }

    #[test]
    fn test_list_endpoints_returns_none_for_sources_without_base_url() {
        // `DummySource` (defined in this module) does not override
        // `base_url`, so it should appear with `None`.
        let mut r = SourceRegistry::new();
        r.register(dummy("nosrc"));
        let endpoints = r.list_endpoints();
        assert_eq!(endpoints, vec![("nosrc".to_string(), None)]);
    }

    #[test]
    fn test_list_endpoints_is_sorted_by_id() {
        let mut r = SourceRegistry::new();
        r.register(dummy("zzz"));
        r.register(dummy("aaa"));
        r.register(dummy("mmm"));
        let ids: Vec<String> = r.list_endpoints().into_iter().map(|(id, _)| id).collect();
        assert_eq!(ids, vec!["aaa".to_string(), "mmm".to_string(), "zzz".to_string()]);
    }
}