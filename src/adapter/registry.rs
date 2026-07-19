use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::ApiAdapter;
use crate::api_error::{ApiError, ApiResult};

pub struct AdapterRegistry {
    by_name: RwLock<HashMap<String, Arc<dyn ApiAdapter>>>,
    by_path: RwLock<HashMap<String, Arc<dyn ApiAdapter>>>,
}

impl AdapterRegistry {
    pub fn new() -> Self {
        Self {
            by_name: RwLock::new(HashMap::new()),
            by_path: RwLock::new(HashMap::new()),
        }
    }

    pub fn register(&self, adapter: Arc<dyn ApiAdapter>) -> ApiResult<()> {
        let mut by_path = self.by_path.write().unwrap();
        for path in adapter.paths() {
            if by_path.contains_key(*path) {
                return Err(ApiError::Internal(format!(
                    "adapter '{}' path conflict: {} already registered",
                    adapter.name(),
                    path
                )));
            }
        }
        for path in adapter.paths() {
            by_path.insert(path.to_string(), adapter.clone());
        }
        self.by_name.write().unwrap()
            .insert(adapter.name().to_string(), adapter.clone());
        Ok(())
    }

    pub fn lookup_by_path(&self, path: &str) -> Option<Arc<dyn ApiAdapter>> {
        self.by_path.read().unwrap().get(path).cloned()
    }

    pub fn lookup_by_name(&self, name: &str) -> Option<Arc<dyn ApiAdapter>> {
        self.by_name.read().unwrap().get(name).cloned()
    }

    pub fn all_paths(&self) -> Vec<String> {
        self.by_path.read().unwrap().keys().cloned().collect()
    }

    pub fn list(&self) -> Vec<String> {
        self.by_name.read().unwrap().keys().cloned().collect()
    }
}

impl Default for AdapterRegistry {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_error::ApiError;
    use crate::canonical::{CanonicalLlmRequest, CanonicalLlmResponse};
    use async_trait::async_trait;
    use bytes::Bytes;

    struct TestAdapter { id: String }
    #[async_trait]
    impl ApiAdapter for TestAdapter {
        fn name(&self) -> &str { &self.id }
        fn paths(&self) -> &[&'static str] { &["/v1/test"] }
        fn parse(&self, _body: &Bytes, _: &axum::http::HeaderMap) -> ApiResult<crate::canonical::CanonicalRequest> {
            unimplemented!()
        }
        fn serialize_response(&self, _resp: &crate::canonical::CanonicalResponse) -> ApiResult<Bytes> {
            unimplemented!()
        }
        fn serialize_stream_event(&self, _event: &crate::canonical::CanonicalStreamEvent)
            -> ApiResult<Option<Bytes>> { unimplemented!() }
        fn response_content_type(&self) -> &'static str { "application/json" }
        fn stream_content_type(&self) -> &'static str { "text/event-stream" }
        fn capabilities(&self) -> crate::capabilities::AdapterCapabilities {
            Default::default()
        }
    }

    #[test]
    fn test_register_and_lookup() {
        let reg = AdapterRegistry::new();
        reg.register(Arc::new(TestAdapter { id: "t".into() })).unwrap();
        assert!(reg.lookup_by_path("/v1/test").is_some());
        assert!(reg.lookup_by_name("t").is_some());
    }

    #[test]
    fn test_path_conflict_rejected() {
        let reg = AdapterRegistry::new();
        reg.register(Arc::new(TestAdapter { id: "a".into() })).unwrap();
        let result = reg.register(Arc::new(TestAdapter { id: "b".into() }));
        assert!(matches!(result, Err(ApiError::Internal(_))));
    }
}