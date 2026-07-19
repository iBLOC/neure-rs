//! neure HTTP server — axum router, handlers, and server state.
//!
//! The module is split into three sub-modules to keep each
//! responsibility focused:
//!
//! * [`error`] — [`ServerError`] enum, its `IntoResponse` impl,
//!   and its `From<NeureError>` conversion.
//! * [`state`] — [`ServerState`] struct and its constructor, which
//!   reads environment variables to select runtimes.
//! * [`handlers`] — all axum handler functions and their request
//!   DTOs.
//!
//! This file (`mod.rs`) re-exports the public types and defines
//! [`create_router`], which wires up all routes into a single
//! axum [`Router`].

mod dispatch;
mod error;
mod handlers;
mod state;

pub use error::ServerError;
pub use handlers::{
    audio_speech, audio_transcriptions, chat_completions, embedding_handler, forecast,
    health_handler, info_handler, list_models, rerank, vision_classify, vision_detect,
    vision_lora_list, vision_lora_register, vision_lora_unregister, vision_pose, vision_segment,
    ChatCompletionsRequest, EmbeddingHttpRequest, ModelList, RerankHttpRequest,
    SpeechRequest, TranscriptionResponse,
};
pub use state::ServerState;

use axum::{
    extract::State,
    routing::{get, post},
    Router,
};
use crate::config::NeureConfig;

/// Build the full axum [`Router`] with all endpoints registered.
///
/// Called from [`embedded.rs`](crate::run_embedded) and also usable
/// by external host processes that want to embed the router in their
/// own axum application.
pub fn create_router(config: NeureConfig) -> Router {
    let state = ServerState::new(config);

    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/info", get(info_handler))
        .route("/v1/models", get(crate::models::handlers::list_models))
        .route("/v1/models/pull", post(crate::models::handlers::pull_model).get(crate::models::handlers::list_pull_jobs))
        .route(
            "/v1/models/pull/{job_id}",
            get(crate::models::handlers::pull_status)
                .delete(crate::models::handlers::cancel_pull_job),
        )
        .route(
            "/v1/models/{engine}/{id}",
            get(crate::models::handlers::get_model)
                .delete(crate::models::handlers::delete_model),
        )
        .route("/v1/catalog/sources", get(crate::models::handlers::list_sources))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/audio/speech", post(audio_speech))
        .route("/v1/audio/transcriptions", post(audio_transcriptions))
        .route("/v1/rerank", post(rerank))
        .route("/v1/embeddings", post(embedding_handler))
        .route("/v1/vision/detect", post(vision_detect))
        .route("/v1/vision/classify", post(vision_classify))
        .route("/v1/vision/segment", post(vision_segment))
        .route("/v1/vision/pose", post(vision_pose))
        .route("/v1/vision/lora/register", post(vision_lora_register))
        .route("/v1/vision/lora/list", get(vision_lora_list))
        .route(
            "/v1/vision/lora/{id}",
            get(vision_lora_list).delete(vision_lora_unregister),
        )
        .route("/v1/messages", post(crate::server::dispatch::adapter_dispatch))
        .route("/v1/forecast", post(forecast))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    // -- ServerError HTTP response tests -----------------------------------

    #[tokio::test]
    async fn test_server_error_bad_request_maps_to_400() {
        let resp =
            ServerError::BadRequest("messages cannot be empty".to_string()).into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["message"], "messages cannot be empty");
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert!(json["error"]["param"].is_null());
    }

    #[tokio::test]
    async fn test_server_error_bad_request_with_param_includes_param_field() {
        let resp = ServerError::BadRequestWithParam(
            "value must be positive".to_string(),
            "top_n".to_string(),
        )
        .into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["message"], "value must be positive");
        assert_eq!(json["error"]["param"], "top_n");
    }

    #[tokio::test]
    async fn test_server_error_not_implemented_maps_to_501() {
        let resp =
            ServerError::NotImplemented("streaming not ready".to_string()).into_response();
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["message"], "streaming not ready");
        assert_eq!(json["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn test_server_error_internal_maps_to_500() {
        let resp = ServerError::Internal("unexpected".to_string()).into_response();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["message"], "unexpected");
    }

    #[test]
    fn test_server_error_display_all_variants() {
        assert_eq!(
            format!("{}", ServerError::BadRequest("x".to_string())),
            "BadRequest: x"
        );
        assert_eq!(
            format!(
                "{}",
                ServerError::BadRequestWithParam("x".to_string(), "p".to_string())
            ),
            "BadRequest: x (param=p)"
        );
        assert_eq!(
            format!("{}", ServerError::NotImplemented("x".to_string())),
            "NotImplemented: x"
        );
        assert_eq!(
            format!("{}", ServerError::Internal("x".to_string())),
            "Internal: x"
        );
    }

    #[test]
    fn test_server_error_from_neure_error_wraps_message() {
        let neure_err = crate::llm::NeureError::not_implemented("legacy path");
        let server_err: ServerError = neure_err.into();
        match server_err {
            ServerError::NotImplemented(m) => assert_eq!(m, "legacy path"),
            other => panic!("expected NotImplemented, got: {other:?}"),
        }
    }

    #[test]
    fn test_server_error_from_invalid_input_maps_to_bad_request() {
        let neure_err = crate::llm::NeureError::invalid_input("bad input");
        let server_err: ServerError = neure_err.into();
        match server_err {
            ServerError::BadRequest(m) => assert_eq!(m, "bad input"),
            other => panic!("expected BadRequest, got: {other:?}"),
        }
    }
}
