---
title: Embedding in Hosts
---

# Embedding in Hosts

`neure` is designed to be embedded into Rust host applications — desktop shells, server-side processes, mobile binaries (via `cargo-ndk`), agent orchestrators. This page documents the integration patterns.

## Common integration: `run_embedded()`

The single API surface for any host:

```rust
use neure::{run_embedded, NeureConfig, NeureEmbedConfig, NeureHandle};

let handle: NeureHandle = run_embedded(NeureEmbedConfig {
    port: 8085,                          // any free port; 8083 is the default
    config: NeureConfig::default(),
}).await?;

// ... host runs its own logic ...

handle.request_shutdown();
handle.join().await;
```

`NeureHandle` owns the axum server's lifetime. Drop the handle without `request_shutdown()` and you leak the bind port — the type system enforces correct lifecycle.

## Pattern 1: Tauri 2 desktop shell

A Tauri 2 app embeds neure as part of its Rust binary. The React/Vue/Svelte front-end talks to neure through a Tauri command bridge that proxies HTTP calls.

**Project structure**:
```
my-tauri-app/
├── src-tauri/
│   ├── Cargo.toml          # path = "../neure"
│   ├── src/
│   │   ├── main.rs        # Tauri Builder::default().setup(|app| { neure::run_embedded(...) })
│   │   └── commands.rs     # Tauri commands: neure_health, neure_list_models, neure_chat
│   ├── tauri.conf.json
│   └── ...
├── src/                    # React / Vue / Svelte front-end
│   ├── lib/
│   │   └── neure.ts        # Tauri command wrappers (invoke('neure_chat', ...))
│   └── ...
├── package.json
└── ...
```

**`src-tauri/Cargo.toml`**:
```toml
[dependencies]
neure = { path = "../../neure", default-features = false, features = ["candle"] }
tauri = { version = "2", features = [] }
tokio = { version = "1", features = ["full"] }
```

**`src-tauri/src/main.rs`** (Tauri 2 setup hook):
```rust
use neure::{run_embedded, NeureConfig, NeureEmbedConfig, NeureHandle};
use tauri::Manager;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let neure_handle = run_embedded(NeureEmbedConfig {
        port: 8085,
        config: NeureConfig::default(),
    }).await?;

    tauri::Builder::default()
        .setup(|app| {
            // Store the handle in the Tauri state so commands can reach it
            app.manage(neure_handle);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            neure_health,
            neure_list_models,
            neure_chat,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");

    Ok(())
}
```

**`src-tauri/src/commands.rs`**:
```rust
use neure::NeureHandle;
use tauri::State;

#[tauri::command]
pub async fn neure_health(_handle: State<'_, NeureHandle>) -> Result<String, String> {
    Ok("ok".to_string())
}

#[tauri::command]
pub async fn neure_list_models(handle: State<'_, NeureHandle>) -> Result<serde_json::Value, String> {
    // call neure's /v1/models
    let resp: serde_json::Value = reqwest::get("http://127.0.0.1:8085/v1/models")
        .await.map_err(|e| e.to_string())?
        .json().await.map_err(|e| e.to_string())?;
    Ok(resp)
}

#[tauri::command]
pub async fn neure_chat(
    model: String, messages: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let resp: serde_json::Value = reqwest::Client::new()
        .post("http://127.0.0.1:8085/v1/chat/completions")
        .json(&serde_json::json!({ "model": model, "messages": messages, "stream": false }))
        .send().await.map_err(|e| e.to_string())?
        .json().await.map_err(|e| e.to_string())?;
    Ok(resp)
}
```

**`src/lib/neure.ts`** (front-end):
```typescript
import { invoke } from '@tauri-apps/api/core';

export async function chat(messages: Message[]): Promise<string> {
  const resp = await invoke<ChatResponse>('neure_chat', {
    model: 'qwen2.5-0.5b',
    messages,
  });
  return resp.choices[0].message.content;
}
```

The Tauri command bridge is the **only** RPC surface the front-end needs to know about — it never talks to neure over HTTP directly (the Tauri shell handles that). This is more secure than exposing neure's HTTP server to the OS.

## Pattern 2: Flutter mobile (via `cargo-ndk`)

A Flutter Android app loads neure as a native library and calls it through a Kotlin shim. The shim uses neure's `cdylib` output:

**Build**:
```bash
cargo ndk \
  --target aarch64-linux-android21 \
  --android-api 21 \
  -- build --release --features candle
# → target/aarch64-linux-android/release/libneure.so
```

**`android/app/build.gradle`** (add to dependencies):
```gradle
implementation fileTree(dir: 'libs', include: ['*.so'])
```

**Kotlin shim** (`android/app/src/main/kotlin/.../NeureBridge.kt`):
```kotlin
class NeureBridge {
    companion object {
        init { System.loadLibrary("neure") }
    }

    external fun startEngine(port: Int, configJson: String): Long
    external fun stopEngine(handle: Long)
    external fun chatCompletions(handle: Long, requestJson: String): String
}
```

**`src/ndk/bridge.c`** (FFI wrapper around neure's `cdylib` surface):
```c
#include <jni.h>
#include "neure.h"  // generated by cbindgen from neure's pub extern "C" surface

JNIEXPORT jlong JNICALL
Java_com_example_NeureBridge_startEngine(JNIEnv *env, jclass cls, jint port, jstring config) {
    const char *cfg = (*env)->GetStringUTFChars(env, config, NULL);
    NeureHandle *handle = neure_run_embedded(port, cfg);
    (*env)->ReleaseStringUTFChars(env, config, cfg);
    return (jlong)handle;
}
```

(For the full FFI bridge you'd need to also wrap the async streaming path; see the [Tauri 2 desktop pattern](#pattern-1-tauri-2-desktop-shell) above for the simpler synchronous-IPC model first.)

## Pattern 3: Server-side process (HTTP gateway)

A server-side process that already has its own main loop can host neure in a background task and proxy requests to it. This is the right pattern when you want to expose neure's capabilities behind a different API (e.g. an internal agent orchestrator).

```rust
use neure::{run_embedded, NeureConfig, NeureEmbedConfig, NeureHandle};
use std::sync::Arc;
use tokio::sync::OnceCell;

static NEURE: OnceCell<Arc<NeureHandle>> = OnceCell::const_new();

async fn ensure_neure_started() -> &'static Arc<NeureHandle> {
    NEURE.get_or_init(|| async {
        let handle = run_embedded(NeureEmbedConfig {
            port: 8085,
            config: NeureConfig::default(),
        }).await.expect("neure failed to start");
        Arc::new(handle)
    }).await
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ... start the host's own HTTP server (e.g. axum on port 7000) ...
    // any handler that needs inference calls ensure_neure_started() then proxies
    // to http://127.0.0.1:8085/v1/*
    Ok(())
}
```

## Pattern 4: Multiple instances (for multi-tenancy / sharded inference)

`neure` can run multiple instances in the same process by giving each its own port and `NeureHandle`. Useful for sharded inference, A/B testing, or per-tenant isolation.

```rust
let handle_a = run_embedded(NeureEmbedConfig { port: 8085, config: NeureConfig::default() }).await?;
let handle_b = run_embedded(NeureEmbedConfig { port: 8086, config: NeureConfig::default() }).await?;

// route to handle_a or handle_b based on tenant / shard key
```

Each handle is independent — no shared state. Resource usage scales with the number of instances.

## Configuration patterns

### Hard-coded config (development)

```rust
let config = NeureConfig::default();  // all env vars override at runtime
```

### Env-driven config (production)

Set `NEURE_*` env vars before launching the host binary; `NeureConfig::from_env()` reads them on startup.

### Programmatic config (tests / dynamic)

```rust
let config = NeureConfig {
    port: 8085,
    host: "127.0.0.1".to_string(),
    llm_runtime: "candle".to_string(),
    device: DeviceSelection::Cpu,
    ..NeureConfig::default()
};
```

See [Environment Variables](/reference/env-vars) for the full `NEURE_*` matrix.

## What NOT to do

- **Don't try to run neure as a standalone binary**. There is no `neure serve` CLI. (If you need a standalone server, use the historical `prefrontal` project which embeds neure that way.)
- **Don't spawn a sidecar process**. The whole point of `run_embedded()` is to avoid the inter-process hop.
- **Don't drop the handle without `request_shutdown() + join()`.** The bind port will leak. The type system is the only thing that prevents this.
- **Don't reach around the type system to share a `NeureHandle` across threads without `Arc<NeureHandle>`**. The handle is `!Send`; wrap it explicitly.

## Next steps

- [Quick Start](/guide/quick-start) — minimal embed example
- [Architecture](/concepts/architecture) — how the data flow wires registries to handlers
- [Environment Variables](/reference/env-vars) — full `NEURE_*` configuration
