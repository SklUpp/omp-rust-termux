//! Per-model catalog metadata consumed by real runtime dispatch: the wire
//! dialect (`api`), endpoint (`baseUrl`), sparse `compat` block and authored
//! `thinking` config must survive `models.json` + `models.yml` loading, with
//! the TS precedence: `models.yml` authors put model keys above provider
//! defaults, while a provider block overriding a bundled `models.json` row
//! wins key by key.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use omp_cli::catalog::Catalog;

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn temp_dir() -> PathBuf {
	let dir = std::env::temp_dir().join(format!(
		"omp-catalog-metadata-{}-{}",
		std::process::id(),
		COUNTER.fetch_add(1, Ordering::Relaxed)
	));
	fs::create_dir_all(&dir).expect("temp dir");
	dir
}

fn write(path: &Path, text: &str) {
	fs::write(path, text).expect("write catalog fixture");
}

/// Bundled-style `models.json` rows plus a `models.yml` provider block: the
/// declared per-model dialect must not be replaced by the provider default,
/// and both compat sources must merge instead of one erasing the other.
#[test]
fn json_rows_keep_declared_dialect_and_merge_provider_compat() {
	let dir = temp_dir();
	let json = dir.join("models.json");
	write(
		&json,
		r#"{
			"ollama": {
				"qwen3": {
					"id": "qwen3",
					"name": "Qwen3",
					"api": "ollama-chat",
					"baseUrl": "http://127.0.0.1:11434",
					"maxTokens": 8192,
					"contextWindow": 131072,
					"input": ["text", "image"],
					"reasoning": true,
					"thinking": {"mode": "effort", "efforts": ["low", "high"], "maxTokens": 4096},
					"compat": {"keepAlive": true, "supportsUsageInStreaming": true, "nested": {"a": 1, "b": 2}},
					"cost": {"input": 1, "output": 2}
				},
				"llama3": {
					"id": "llama3",
					"api": "anthropic-messages",
					"baseUrl": "https://proxy.example/v1"
				}
			}
		}"#,
	);
	let yml = dir.join("models.yml");
	write(
		&yml,
		"providers:\n  ollama:\n    baseUrl: http://gateway.example:8080\n    api: openai-completions\n    compat:\n      keepAlive: false\n      supportsUsageInStreaming: false\n      nested:\n        b: 9\n        c: 3\n    models:\n      - id: qwen3\n        api: ollama-chat\n        baseUrl: http://127.0.0.1:11434\n",
	);
	let catalog =
		Catalog::load_from(&[json], &yml).expect("json rows win over yml model lists");

	// Per-model dialects stay distinct: the provider `api` never masks them.
	assert_eq!(catalog.model_api("ollama", "qwen3"), Some("ollama-chat"));
	assert_eq!(catalog.model_api("ollama", "llama3"), Some("anthropic-messages"));
	assert_eq!(
		catalog.model_base_url("ollama", "qwen3"),
		Some("http://127.0.0.1:11434")
	);
	assert_eq!(
		catalog.model_base_url("ollama", "llama3"),
		Some("https://proxy.example/v1")
	);
	let connection = catalog.connections.get("ollama").expect("connection");
	assert_eq!(connection.api.as_deref(), Some("openai-completions"));
	assert_eq!(connection.base_url.as_deref(), Some("http://gateway.example:8080"));

	// The `models.yml` provider block is a user override for bundled rows:
	// nested records merge, scalar collisions go to the override.
	let compat = catalog.model_compat("ollama", "qwen3").expect("compat");
	assert_eq!(compat["keepAlive"], serde_json::json!(false));
	assert_eq!(compat["supportsUsageInStreaming"], serde_json::json!(false));
	assert_eq!(compat["nested"], serde_json::json!({"a": 1, "b": 9, "c": 3}));
	assert_eq!(
		catalog.supports_usage_in_streaming("ollama", "qwen3"),
		Some(false),
		"the usage flag tracks the merged compat block"
	);
	// A row without its own block inherits the provider override.
	assert_eq!(
		catalog.model_compat("ollama", "llama3"),
		Some(&serde_json::json!({
			"keepAlive": false,
			"supportsUsageInStreaming": false,
			"nested": {"b": 9, "c": 3},
		}))
	);

	// Authored thinking metadata survives verbatim for adapters.
	assert_eq!(
		catalog.model_thinking("ollama", "qwen3"),
		Some(&serde_json::json!({"mode": "effort", "efforts": ["low", "high"], "maxTokens": 4096}))
	);
	assert_eq!(catalog.model_thinking("ollama", "llama3"), None);

	// Existing listing behavior is unchanged.
	let info = &catalog.providers["ollama"]["qwen3"];
	assert_eq!(info.name.as_deref(), Some("Qwen3"));
	assert_eq!(info.max_tokens, Some(8192));
	assert_eq!(catalog.context_window("ollama", "qwen3"), Some(131072));
	assert_eq!(
		catalog.model_input("ollama", "qwen3"),
		Some(&["text".to_string(), "image".to_string()][..])
	);
	assert_eq!(catalog.supports_images("ollama", "qwen3"), Some(true));
	assert_eq!(info.efforts.as_deref(), Some(&["low".to_string(), "high".to_string()][..]));
	let cost = info.cost.as_ref().expect("cost");
	assert_eq!(cost["input"], serde_json::json!(1));
	assert_eq!(cost["cacheRead"], serde_json::json!(0));
	fs::remove_dir_all(dir).ok();
}

/// `models.yml`-authored models use the TS custom-provider precedence: the
/// provider block is the base, the model's own keys win.
#[test]
fn yml_models_keep_model_metadata_above_provider_defaults() {
	let dir = temp_dir();
	let yml = dir.join("models.yml");
	write(
		&yml,
		"providers:\n  local:\n    baseUrl: http://127.0.0.1:8080\n    api: openai-completions\n    compat:\n      supportsUsageInStreaming: false\n      disableStrictTools: false\n      nested:\n        x: 1\n    models:\n      - id: alpha\n        api: anthropic-messages\n        baseUrl: http://127.0.0.1:9000\n        thinking:\n          mode: budget\n          efforts:\n            - low\n            - medium\n          maxTokens: 2048\n        compat:\n          supportsUsageInStreaming: true\n          nested:\n            y: 2\n      - id: beta\n",
	);
	let catalog = Catalog::load_from(&[dir.join("missing.json")], &yml).expect("catalog");

	assert_eq!(catalog.model_api("local", "alpha"), Some("anthropic-messages"));
	assert_eq!(
		catalog.model_base_url("local", "alpha"),
		Some("http://127.0.0.1:9000")
	);
	// No declared per-model value: consumers fall back to the connection.
	assert_eq!(catalog.model_api("local", "beta"), None);
	assert_eq!(catalog.model_base_url("local", "beta"), None);
	assert_eq!(
		catalog.connections["local"].api.as_deref(),
		Some("openai-completions")
	);

	let alpha = catalog.model_compat("local", "alpha").expect("alpha compat");
	assert_eq!(alpha["supportsUsageInStreaming"], serde_json::json!(true));
	assert_eq!(alpha["disableStrictTools"], serde_json::json!(false));
	assert_eq!(alpha["nested"], serde_json::json!({"x": 1, "y": 2}));
	let beta = catalog.model_compat("local", "beta").expect("beta compat");
	assert_eq!(beta["supportsUsageInStreaming"], serde_json::json!(false));
	assert_eq!(catalog.supports_usage_in_streaming("local", "alpha"), Some(true));
	assert_eq!(catalog.supports_usage_in_streaming("local", "beta"), Some(false));

	assert_eq!(
		catalog.model_thinking("local", "alpha"),
		Some(&serde_json::json!({"mode": "budget", "efforts": ["low", "medium"], "maxTokens": 2048}))
	);
	assert_eq!(
		catalog.providers["local"]["alpha"].efforts.as_deref(),
		Some(&["low".to_string(), "medium".to_string()][..])
	);
	fs::remove_dir_all(dir).ok();
}
