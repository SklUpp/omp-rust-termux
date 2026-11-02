use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use serde_json::json;

use super::*;

static COUNTER: AtomicU64 = AtomicU64::new(0);

struct TempDir(PathBuf);

impl TempDir {
	fn new(label: &str) -> Self {
		let nonce = COUNTER.fetch_add(1, Ordering::Relaxed);
		let path = std::env::temp_dir().join(format!("omp-c5-{label}-{}-{nonce}", std::process::id()));
		fs::create_dir_all(&path).expect("create temp dir");
		Self(path)
	}

	fn path(&self) -> &Path {
		&self.0
	}

	fn write(&self, relative: &str, content: &str) -> PathBuf {
		let path = self.0.join(relative);
		if let Some(parent) = path.parent() {
			fs::create_dir_all(parent).expect("create parent");
		}
		fs::write(&path, content).expect("write fixture");
		path
	}
}

impl Drop for TempDir {
	fn drop(&mut self) {
		let _ = fs::remove_dir_all(&self.0);
	}
}

fn set_old_mtime(path: &Path, age: Duration) {
	let file = fs::OpenOptions::new().write(true).open(path).expect("open for mtime");
	file.set_modified(SystemTime::now() - age).expect("set modified");
}

fn run_ttsr_capture(opts: &TtsrOptions, cwd: &Path) -> (u8, String, String) {
	let mut out = Vec::new();
	let mut err = Vec::new();
	let code = run_ttsr(opts, cwd, &mut out, &mut err).expect("ttsr handler");
	(
		code,
		String::from_utf8_lossy(&out).into_owned(),
		String::from_utf8_lossy(&err).into_owned(),
	)
}

fn sample_rule(dir: &TempDir) -> PathBuf {
	dir.write(
		"no-any.md",
		"---\ncondition: '\\bany\\b'\nscope: 'tool:edit(*.ts)'\ndescription: Avoid any\n---\nUse unknown, not any.\n",
	)
}

#[test]
fn ttsr_isolated_test_reports_triggered_rule_json() {
	let dir = TempDir::new("ttsr-json");
	let rule = sample_rule(&dir);
	let opts = TtsrOptions {
		action:       Some("test".to_string()),
		snippet:      Some("let value: any = 1;".to_string()),
		file:         None,
		rule:         Some(rule),
		source:       Some("tool".to_string()),
		tool:         Some("edit".to_string()),
		path:         Some("src/a.ts".to_string()),
		verbose:      false,
		json:         true,
		no_gitignore: false,
		max_bytes:    None,
	};
	let (code, stdout, _) = run_ttsr_capture(&opts, dir.path());
	assert_eq!(code, 0);
	let report: Value = serde_json::from_str(stdout.trim()).expect("json report");
	assert_eq!(report["source"], "tool");
	assert_eq!(report["tool"], "edit");
	assert_eq!(report["filePath"], "src/a.ts");
	assert_eq!(report["evaluated"], 1);
	assert_eq!(report["triggered"][0]["name"], "no-any");
	assert_eq!(report["triggered"][0]["matched"]["regex"][0], "\\bany\\b");
	assert_eq!(report["triggered"][0]["sourceProvider"], "ttsr-cli");
	assert_eq!(report["notTriggered"], json!([]));
	assert!(report.get("inferenceNote").is_none());
}

#[test]
fn ttsr_isolated_test_renders_triggered_and_defined_sections() {
	let dir = TempDir::new("ttsr-text");
	let rule = sample_rule(&dir);
	let opts = TtsrOptions {
		action:       Some("test".to_string()),
		snippet:      Some("aTS".to_string()),
		file:         None,
		rule:         Some(rule),
		source:       Some("text".to_string()),
		tool:         None,
		path:         None,
		verbose:      true,
		json:         false,
		no_gitignore: false,
		max_bytes:    None,
	};
	let (code, stdout, _) = run_ttsr_capture(&opts, dir.path());
	assert_eq!(code, 1, "isolated rule with no trigger exits 1");
	assert!(stdout.contains("TTSR test — source=text"), "{stdout}");
	assert!(stdout.contains("No rules triggered. (evaluated 1)"), "{stdout}");
	assert!(stdout.contains("Not triggered (1)"), "{stdout}");
	assert!(stdout.contains("condition: /\\bany\\b/"), "{stdout}");
}

#[test]
fn ttsr_invalid_source_and_rule_gaps_are_exact() {
	let dir = TempDir::new("ttsr-errors");
	let opts = TtsrOptions {
		action:       Some("test".to_string()),
		snippet:      Some("x".to_string()),
		file:         None,
		rule:         None,
		source:       Some("bogus".to_string()),
		tool:         None,
		path:         None,
		verbose:      false,
		json:         false,
		no_gitignore: false,
		max_bytes:    None,
	};
	let error = run_ttsr(&opts, dir.path(), &mut Vec::new(), &mut Vec::new()).expect_err("invalid source");
	assert_eq!(
		error.to_string(),
		"Invalid --source: bogus. Expected one of: text, thinking, tool"
	);

	let rule = dir.write("plain.md", "No frontmatter here.\n");
	let opts = TtsrOptions {
		rule: Some(rule),
		source: Some("text".to_string()),
		..opts
	};
	let error = run_ttsr(&opts, dir.path(), &mut Vec::new(), &mut Vec::new()).expect_err("no condition");
	assert_eq!(
		error.to_string(),
		"Rule \"plain\" has no usable TTSR condition. Add a `condition` (regex) or `astCondition` (ast-grep pattern) to its frontmatter."
	);
}

#[test]
fn ttsr_scan_counts_rule_matches() {
	let dir = TempDir::new("ttsr-scan");
	let rule = sample_rule(&dir);
	dir.write("src/a.ts", "let value: any = 1;\n");
	dir.write("src/b.ts", "let value: unknown = 1;\n");
	dir.write("notes.txt", "any\n");
	let scan_dir = dir.path().join("src");
	let opts = TtsrOptions {
		action:       Some("scan".to_string()),
		snippet:      Some(scan_dir.display().to_string()),
		file:         None,
		rule:         Some(rule),
		source:       None,
		tool:         None,
		path:         None,
		verbose:      true,
		json:         false,
		no_gitignore: true,
		max_bytes:    None,
	};
	let (code, stdout, stderr) = run_ttsr_capture(&opts, dir.path());
	assert_eq!(code, 0, "{stderr}");
	assert!(stdout.contains("TTSR scan — directory="), "{stdout}");
	assert!(stdout.contains("Found violations/matches: (1 matches across 1 files)"), "{stdout}");
	assert!(stdout.contains("a.ts"), "{stdout}");
	assert!(!stdout.contains("b.ts"), "{stdout}");
}

#[test]
fn gallery_state_aliases_and_errors_are_exact() {
	assert_eq!(
		parse_gallery_states(&["streaming args".into(), "in progress".into(), "done".into(), "failed".into()])
			.expect("known states"),
		vec![
			GalleryState::Streaming,
			GalleryState::Progress,
			GalleryState::Success,
			GalleryState::Error
		]
	);
	let error = parse_gallery_states(&["nope".into()]).expect_err("invalid state");
	assert_eq!(
		error.to_string(),
		"Invalid --state 'nope'. Valid values: streaming, streaming args, progress, in progress, success, done, error, failed"
	);
}

#[test]
fn gallery_covers_every_runtime_tool() {
	let mut expected: Vec<String> = crate::runtime::BUILTIN_TOOLS
		.iter()
		.map(|(name, _)| (*name).to_string())
		.chain(crate::runtime::DELEGATION_TOOLS.iter().map(|name| (*name).to_string()))
		.collect();
	expected.extend(["checkpoint", "rewind", "goal", "think"].map(str::to_string));
	expected.sort();
	expected.dedup();
	assert_eq!(gallery_known_tools(), expected, "gallery table must cover the Rust tool roster");
}

#[test]
fn gallery_unknown_tool_and_plain_output() {
	let opts = GalleryOptions {
		tool:     Some("nope".to_string()),
		states:   vec!["done".to_string()],
		width:    Some(80),
		expanded: false,
		plain:    true,
	};
	let mut out = Vec::new();
	assert_eq!(run_gallery(&opts, &mut out).expect("gallery"), 0);
	let stdout = String::from_utf8(out).expect("utf8");
	assert!(stdout.starts_with("Unknown tool 'nope'. Known tools: "), "{stdout}");
	assert!(stdout.contains("read"), "{stdout}");

	let opts = GalleryOptions {
		tool:     Some("read".to_string()),
		states:   vec!["done".to_string(), "failed".to_string()],
		width:    Some(10),
		expanded: true,
		plain:    true,
	};
	let mut out = Vec::new();
	assert_eq!(run_gallery(&opts, &mut out).expect("gallery"), 0);
	let stdout = String::from_utf8(out).expect("utf8");
	assert!(!stdout.contains('\x1b'), "plain output must carry no ANSI codes");
	assert!(stdout.contains("── read "), "{stdout}");
	assert!(stdout.contains("· done"), "{stdout}");
	assert!(stdout.contains("✗ read: Error: operation failed"), "{stdout}");

	let opts = GalleryOptions { plain: false, ..opts };
	let mut out = Vec::new();
	run_gallery(&opts, &mut out).expect("gallery");
	assert!(String::from_utf8(out).expect("utf8").contains("\x1b["));
}

#[test]
fn grievances_missing_empty_and_rows() {
	let dir = TempDir::new("grievances");
	let opts = GrievancesOptions { limit: 20, tool: None, json: false };

	let mut out = Vec::new();
	list_grievances_in(&dir.path().join("missing.db"), &opts, &mut out).expect("missing db");
	assert_eq!(
		String::from_utf8(out).expect("utf8"),
		"No grievances database found. Auto-QA has not recorded any reports yet (or was disabled).\n"
	);

	let db_path = dir.path().join("autoqa.db");
	let connection = rusqlite::Connection::open(&db_path).expect("open db");
	connection
		.execute_batch(
			"CREATE TABLE grievances (id INTEGER PRIMARY KEY, model TEXT, version TEXT, tool TEXT, report TEXT);",
		)
		.expect("schema");

	let mut out = Vec::new();
	list_grievances_in(&db_path, &opts, &mut out).expect("empty db");
	assert_eq!(String::from_utf8(out).expect("utf8"), "No grievances recorded yet.\n");

	let json_opts = GrievancesOptions { limit: 20, tool: None, json: true };
	let mut out = Vec::new();
	list_grievances_in(&db_path, &json_opts, &mut out).expect("empty json");
	assert_eq!(String::from_utf8(out).expect("utf8"), "[]\n");

	connection
		.execute(
			"INSERT INTO grievances (id, model, version, tool, report) VALUES (1, 'model-x', '1.2.3', 'bash', 'boom')",
			[],
		)
		.expect("insert");
	let mut out = Vec::new();
	list_grievances_in(&db_path, &opts, &mut out).expect("rows");
	assert_eq!(
		String::from_utf8(out).expect("utf8"),
		"#1 bash (model-x v1.2.3)\n  boom\n\nShowing 1 most recent\n"
	);

	let mut out = Vec::new();
	list_grievances_in(&db_path, &json_opts, &mut out).expect("rows json");
	let rows: Value = serde_json::from_slice(&out).expect("json rows");
	assert_eq!(rows[0]["id"], 1);
	assert_eq!(rows[0]["tool"], "bash");
	assert_eq!(rows[0]["report"], "boom");
}

fn gc_json(opts: &GcOptions, cwd: &Path) -> (u8, Value, String) {
	let mut out = Vec::new();
	let mut err = Vec::new();
	let code = run_gc(opts, cwd, &mut out, &mut err).expect("gc handler");
	let value = serde_json::from_slice(&out).expect("gc json");
	(code, value, String::from_utf8_lossy(&err).into_owned())
}

#[test]
fn gc_blob_dry_run_counts_referenced_and_unreferenced() {
	let dir = TempDir::new("gc-blobs");
	let agent = dir.path().join("agent");
	let referenced = "a".repeat(64);
	let orphan = "b".repeat(64);
	let referenced_file = dir.write(&format!("agent/blobs/{referenced}"), "ref!!");
	let orphan_file = dir.write(&format!("agent/blobs/{orphan}"), "orphan!");
	dir.write(
		"agent/sessions/project/s1.jsonl",
		&format!("{{\"type\":\"message\",\"message\":{{\"content\":\"blob:sha256:{referenced}\"}}}}\n"),
	);
	set_old_mtime(&referenced_file, Duration::from_secs(3600));
	set_old_mtime(&orphan_file, Duration::from_secs(3600));

	let opts = GcOptions {
		json:                    true,
		agent_dir:               Some(agent),
		blobs:                   true,
		archive:                 false,
		wal:                     false,
		cold_archive_after_days: None,
		retain_newest_global:    None,
		retain_newest_per_cwd:   None,
	};
	let (code, value, stderr) = gc_json(&opts, dir.path());
	assert_eq!(code, 0, "{stderr}");
	assert_eq!(value["apply"], false);
	assert_eq!(value["blobs"]["referenced"], 1);
	assert_eq!(value["blobs"]["candidates"], 2);
	assert_eq!(value["blobs"]["wouldDelete"], 1);
	assert_eq!(value["blobs"]["deleted"], 0);
	assert_eq!(value["blobs"]["bytes"], "orphan!".len() as u64);
	assert!(orphan_file.exists(), "dry-run must not delete blobs");
	assert!(value["lockPath"].as_str().is_some_and(|path| path.ends_with("gc.lock")));
}

#[test]
fn gc_archive_dry_run_counts_cold_complete_sessions() {
	let dir = TempDir::new("gc-archive");
	let agent = dir.path().join("agent");
	let session = dir.write(
		"agent/sessions/project/s1.jsonl",
		"{\"type\":\"session\",\"id\":\"s1\",\"cwd\":\"/tmp/project\"}\n{\"type\":\"message\",\"message\":{\"role\":\"assistant\",\"stopReason\":\"stop\",\"content\":[]}}\n",
	);
	set_old_mtime(&session, Duration::from_secs(60 * 86_400));

	let opts = GcOptions {
		json:                    true,
		agent_dir:               Some(agent),
		blobs:                   false,
		archive:                 true,
		wal:                     false,
		cold_archive_after_days: Some(30),
		retain_newest_global:    Some(0),
		retain_newest_per_cwd:   Some(0),
	};
	let (code, value, stderr) = gc_json(&opts, dir.path());
	assert_eq!(code, 0, "{stderr}");
	assert_eq!(value["archive"]["scanned"], 1);
	assert_eq!(value["archive"]["skippedActive"], 0);
	assert_eq!(value["archive"]["wouldArchive"], 1);
	assert_eq!(value["archive"]["archived"], 0);
	assert!(session.exists(), "dry-run must keep session logs");
}

#[test]
fn gc_wal_dry_run_reports_wal_bytes() {
	let dir = TempDir::new("gc-wal");
	let agent = dir.path().join("agent");
	let db = dir.write("agent/history.db", "");
	dir.write("agent/history.db-wal", "wal-bytes");
	let opts = GcOptions {
		json:                    true,
		agent_dir:               Some(agent),
		blobs:                   false,
		archive:                 false,
		wal:                     true,
		cold_archive_after_days: None,
		retain_newest_global:    None,
		retain_newest_per_cwd:   None,
	};
	let (code, value, stderr) = gc_json(&opts, dir.path());
	assert_eq!(code, 0, "{stderr}");
	assert_eq!(value["wal"]["walBytes"], "wal-bytes".len() as u64);
	assert_eq!(value["wal"]["wouldCheckpoint"], true);
	assert_eq!(value["wal"]["checkpointed"], false);
	assert_eq!(value["wal"]["databases"].as_array().map(Vec::len), Some(2));
	assert!(db.exists());
}

#[test]
fn setup_component_errors_are_exact() {
	let dir = TempDir::new("setup-errors");
	let mut out = Vec::new();
	let mut err = Vec::new();
	let opts = SetupOptions { component: None, check: true, json: false };
	let error = run_setup(&opts, dir.path(), &mut out, &mut err).expect_err("missing component");
	assert_eq!(
		error.to_string(),
		"setup --check/--json requires a COMPONENT (python|speech)"
	);

	let opts = SetupOptions { component: None, check: false, json: false };
	let error = run_setup(&opts, dir.path(), &mut out, &mut err).expect_err("headless wizard");
	assert_eq!(error.to_string(), "omp setup requires an interactive TTY.");

	let opts = SetupOptions { component: Some("bogus".to_string()), check: false, json: false };
	let error = run_setup(&opts, dir.path(), &mut out, &mut err).expect_err("unknown component");
	assert_eq!(error.to_string(), "Unknown component: bogus\nValid components: python, speech");
}

#[test]
fn setup_python_probe_rejects_missing_interpreter() {
	let dir = TempDir::new("setup-python");
	let runtime = tokio::runtime::Builder::new_current_thread()
		.enable_all()
		.build()
		.expect("runtime");
	let (available, path) = runtime.block_on(probe_python(
		dir.path(),
		Some("/nonexistent/python-c5"),
		&dir.path().join("python-env"),
	));
	assert!(!available);
	assert!(path.is_none());
}

#[test]
fn setup_speech_probes_model_and_runtime_cache() {
	let dir = TempDir::new("setup-speech");
	let tiny = dir.path().join("tiny-models");
	let stt_dir = "tiny-models/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8";
	// Empty files and directories must not count as cached assets.
	dir.write(&format!("{stt_dir}/encoder.int8.onnx"), "");
	for file in ["decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"] {
		dir.write(&format!("{stt_dir}/{file}"), "x");
	}
	fs::create_dir_all(dir.path().join(stt_dir).join("dummy.int8.onnx")).expect("dummy dir");
	assert!(!stt_model_cached(&tiny, "parakeet"));
	dir.write(&format!("{stt_dir}/encoder.int8.onnx"), "x");
	assert!(stt_model_cached(&tiny, "parakeet"));
	assert!(!stt_model_cached(&tiny, "fast"));
	assert!(!tts_model_cached(&tiny, "kokoro"));

	let kokoro = "tiny-models/onnx-community/Kokoro-82M-v1.0-ONNX/kokoro.onnx";
	dir.write(kokoro, "");
	assert!(!tts_model_cached(&tiny, "kokoro"));
	dir.write(kokoro, "x");
	dir.write(
		"tts-runtime/kokoro-1.2.1/node_modules/kokoro-js/package.json",
		"{\"version\":\"1.2.1\"}",
	);
	let cache_root = tiny.parent().expect("cache root");
	assert!(tts_model_cached(&tiny, "kokoro"));
	assert!(tts_runtime_cached(&tiny));
	assert!(cache_root.exists());
}

#[test]
fn preview_and_scope_parsing_match_cli_contract() {
	assert_eq!(preview_snippet("  a\n\n b  c "), "a b c");
	let long = "x".repeat(100);
	assert_eq!(preview_snippet(&long).chars().count(), 78);
	assert!(preview_snippet(&long).ends_with('…'));

	let scope = parse_tool_scope_token("tool:edit(*.ts)").expect("tool scope");
	assert_eq!(scope.tool_name.as_deref(), Some("edit"));
	assert_eq!(scope.path_glob.as_deref(), Some("*.ts"));
	assert!(parse_tool_scope_token("tool:").is_none());
	assert!(parse_tool_scope_token("bad token").is_none());

	let plan = build_scope(&Rule {
		name:          "r".to_string(),
		path:          PathBuf::from("r.md"),
		content:       String::new(),
		globs:         None,
		always_apply:  false,
		description:   None,
		condition:     None,
		ast_condition: None,
		scope:         None,
		provider:      "native".to_string(),
	});
	assert!(plan.allow_text && plan.allow_any_tool && !plan.allow_thinking);
}

#[test]
fn setup_speech_json_and_check_reports_match_contract() {
	let dir = TempDir::new("setup-speech-cli");
	let tiny = dir.path().join("tiny-models");
	for file in ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"] {
		dir.write(
			&format!("tiny-models/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/{file}"),
			"x",
		);
	}

	let json_opts = SetupOptions { component: Some("speech".to_string()), check: false, json: true };
	let mut out = Vec::new();
	let code = run_setup_speech_with_cache(&json_opts, dir.path(), &tiny, &mut out).expect("speech json");
	assert_eq!(code, 1, "cache presence is not execution readiness");
	let report: Value = serde_json::from_slice(&out).expect("speech json report");
	assert_eq!(report["Speech-to-Text model"]["cached"], true);
	assert_eq!(report["Speech-to-Text model"]["ready"], false);
	assert_eq!(
		report["Speech-to-Text model"]["status"],
		"parakeet — cached; execution runtime not verified"
	);
	assert_eq!(report["Text-to-Speech model"]["cached"], false);
	assert_eq!(report["Text-to-Speech model"]["ready"], false);
	assert_eq!(
		report["Text-to-Speech model"]["status"],
		"kokoro — model/runtime not installed"
	);

	dir.write("tiny-models/onnx-community/Kokoro-82M-v1.0-ONNX/kokoro.onnx", "x");
	dir.write(
		"tts-runtime/kokoro-1.2.1/node_modules/kokoro-js/package.json",
		"{\"version\":\"1.2.1\"}",
	);
	let check_opts = SetupOptions { component: Some("speech".to_string()), check: true, json: false };
	let mut out = Vec::new();
	let code = run_setup_speech_with_cache(&check_opts, dir.path(), &tiny, &mut out).expect("speech check");
	assert_eq!(code, 1, "native build has no verified execution backend");
	let text = String::from_utf8(out).expect("utf8");
	assert!(text.starts_with("Speech dependencies:\n"), "{text}");
	assert!(text.contains("[missing] Speech-to-Text model: parakeet — cached; execution runtime not verified"), "{text}");
	assert!(text.contains("[missing] Text-to-Speech model: kokoro — cached; execution runtime not verified"), "{text}");
	assert!(!text.contains("[ok]"), "{text}");
}

#[test]
fn gc_text_renders_dry_run_lines() {
	let result = json!({
		"agentDir": "/tmp/agent",
		"apply": false,
		"lockPath": "/tmp/agent/gc.lock",
		"blobs": { "referenced": 2, "candidates": 3, "wouldDelete": 1, "deleted": 0, "bytes": 7, "errors": [] },
		"archive": {
			"scanned": 4,
			"skippedActive": 2,
			"keptNewestGlobal": 0,
			"keptNewestPerCwd": 0,
			"wouldArchive": 1,
			"archived": 0,
			"historyRowsDeleted": 0,
			"statsRowsDeleted": 0,
			"ftsRebuilt": false,
			"errors": [],
		},
		"wal": { "databases": [], "walBytes": 0, "wouldCheckpoint": false, "checkpointed": false },
	});
	assert_eq!(
		gc_text(&result),
		concat!(
			"GC dry-run (/tmp/agent)\n",
			"blobs: 0/1 files, 7 B, 2 refs\n",
			"sessions: 0/1 archived, 0 history rows and 0 stats rows removed\n",
			"sessions skipped active: 2\n",
			"wal: checkpoint dry-run, 0 B across 0 dbs\n",
		)
	);
}

fn write_gzip(path: &Path, content: &str) {
	use std::io::Write as _;
	use flate2::{Compression, write::GzEncoder};
	let file = fs::File::create(path).expect("create gzip fixture");
	let mut encoder = GzEncoder::new(file, Compression::best());
	encoder.write_all(content.as_bytes()).expect("gzip fixture");
	encoder.finish().expect("finish gzip fixture");
}

#[test]
fn gc_blob_refs_scan_compressed_and_backup_jsonl() {
	let dir = TempDir::new("gc-blob-refs");
	let agent = dir.path().join("agent");
	let a = "a".repeat(64);
	let b = "b".repeat(64);
	dir.write(
		"agent/sessions/project/s1.jsonl",
		"{\"type\":\"session\",\"id\":\"s1\",\"cwd\":\"/tmp/project\"}\n",
	);
	dir.write(
		"agent/sessions/project/s1.jsonl.1700000000.bak",
		&format!("{{\"blob\":\"BLOB:SHA256:{}\"}}\n", b.to_uppercase()),
	);
	let gz_path = dir.path().join("agent/archive/sessions/a.jsonl.gz");
	fs::create_dir_all(gz_path.parent().expect("gz parent")).expect("gz dir");
	write_gzip(&gz_path, &format!("{{\"blob\":\"blob:sha256:{a}\"}}\n"));

	let a_file = dir.write(&format!("agent/blobs/{a}"), "aaaa");
	let b_file = dir.write(&format!("agent/blobs/{b}.json"), "bb");
	set_old_mtime(&a_file, Duration::from_secs(3600));
	set_old_mtime(&b_file, Duration::from_secs(3600));

	let opts = GcOptions {
		json:                    true,
		agent_dir:               Some(agent),
		blobs:                   true,
		archive:                 false,
		wal:                     false,
		cold_archive_after_days: None,
		retain_newest_global:    None,
		retain_newest_per_cwd:   None,
	};
	let (code, value, stderr) = gc_json(&opts, dir.path());
	assert_eq!(code, 0, "{stderr}");
	assert_eq!(value["blobs"]["referenced"], 2);
	assert_eq!(value["blobs"]["candidates"], 2);
	assert_eq!(value["blobs"]["wouldDelete"], 0);
}

#[test]
fn gc_archive_uses_tail_status_and_nested_liveness() {
	let dir = TempDir::new("gc-session-status");
	let agent = dir.path().join("agent");
	let header = "{\"type\":\"session\",\"id\":\"s\",\"cwd\":\"/tmp/project\"}\n";
	let interrupted = dir.write(
		"agent/sessions/project/interrupted.jsonl",
		&format!(
			"{header}{{\"type\":\"message\",\"message\":{{\"role\":\"assistant\",\"stopReason\":\"length\",\"content\":[]}}}}\n"
		),
	);
	let aborted = dir.write(
		"agent/sessions/project/aborted.jsonl",
		&format!(
			"{header}{{\"type\":\"message\",\"message\":{{\"role\":\"assistant\",\"stopReason\":\"aborted\",\"content\":[]}}}}\n"
		),
	);
	let nested_parent = dir.write(
		"agent/sessions/project/complete-nested.jsonl",
		&format!(
			"{header}{{\"type\":\"message\",\"message\":{{\"role\":\"assistant\",\"stopReason\":\"stop\",\"content\":[]}}}}\n"
		),
	);
	dir.write(
		"agent/sessions/project/complete-nested/nested.jsonl",
		&format!(
			"{header}{{\"type\":\"message\",\"message\":{{\"role\":\"user\",\"content\":\"hi\"}}}}\n"
		),
	);
	// Invalid header is skipped by listSessionsReadOnly; root-level logs are not
	// part of a project directory either.
	dir.write("agent/sessions/project/invalid.jsonl", "{\"type\":\"message\"}\n");
	dir.write("agent/sessions/root.jsonl", header);
	for path in [&interrupted, &aborted, &nested_parent] {
		set_old_mtime(path, Duration::from_secs(60 * 86_400));
	}

	let opts = GcOptions {
		json:                    true,
		agent_dir:               Some(agent),
		blobs:                   false,
		archive:                 true,
		wal:                     false,
		cold_archive_after_days: Some(30),
		retain_newest_global:    Some(0),
		retain_newest_per_cwd:   Some(0),
	};
	let (code, value, stderr) = gc_json(&opts, dir.path());
	assert_eq!(code, 0, "{stderr}");
	assert_eq!(value["archive"]["scanned"], 3);
	assert_eq!(value["archive"]["skippedActive"], 2);
	assert_eq!(value["archive"]["wouldArchive"], 1);
}

#[test]
fn ttsr_builtin_defaults_are_embedded_from_ts_assets() {
	let rules = builtin_default_rules();
	assert_eq!(rules.len(), 27);
	let no_any = rules.iter().find(|rule| rule.name == "ts-no-any").expect("ts-no-any");
	assert_eq!(no_any.provider, "builtin-defaults");
	assert_eq!(no_any.path.to_string_lossy(), "builtin-defaults:ts-no-any.md");
	assert!(no_any.condition.as_ref().is_some_and(|items| !items.is_empty()));
	assert!(no_any.scope.as_ref().is_some_and(|items| items.iter().any(|scope| scope.contains("tool:edit"))));
}

#[test]
fn terminal_escaping_neutralizes_control_sequences() {
	let raw = "safe\n\u{1b}]52;c;base64\u{7}\u{9b}31m";
	let escaped = escape_terminal_text(raw, false);
	assert!(!escaped.contains('\u{1b}'), "{escaped}");
	assert!(!escaped.contains('\u{7}'), "{escaped}");
	assert!(!escaped.contains('\u{9b}'), "{escaped}");
	assert!(!escaped.contains('\n'), "{escaped}");
	assert!(!escaped.chars().any(char::is_control), "{escaped}");

	let kept = escape_terminal_text("a\nb\tc\u{1b}", true);
	assert!(kept.contains('\n'), "newlines preserved: {kept}");
	assert!(kept.contains('\t'), "tabs preserved: {kept}");
	assert!(!kept.contains('\u{1b}'), "other controls still escaped: {kept}");

	let detail = json!({
		"name": "rule\u{1b}[31m",
		"sourceProvider": "native\u{1b}]0;t\u{7}",
		"defined": { "regex": ["a\u{1b}[32m"], "ast": [] },
		"skippedAst": null,
	});
	let mut out = Vec::new();
	render_rule_detail(&mut out, &detail, false).expect("detail");
	let text = String::from_utf8(out).expect("utf8");
	assert!(!text.contains('\u{1b}'), "{text}");
	assert!(!text.contains('\u{7}'), "{text}");
	assert!(text.contains("rule"), "{text}");
}

#[test]
fn grievance_human_output_escapes_terminal_controls() {
	let dir = TempDir::new("grievance-escape");
	let db = dir.path().join("autoqa.db");
	let connection = rusqlite::Connection::open(&db).expect("db");
	connection
		.execute_batch(
			"CREATE TABLE grievances (id INTEGER PRIMARY KEY, model TEXT, version TEXT, tool TEXT, report TEXT);",
		)
		.expect("schema");
	connection
		.execute(
			"INSERT INTO grievances (id, model, version, tool, report) VALUES (1, 'm', '1', 'bash', ?1)",
			rusqlite::params!["line\u{1b}]52;c;x\nsecond\u{7}"],
		)
		.expect("row");
	drop(connection);
	let opts = GrievancesOptions { limit: 20, tool: None, json: false };
	let mut out = Vec::new();
	list_grievances_in(&db, &opts, &mut out).expect("list");
	let text = String::from_utf8(out).expect("utf8");
	assert!(!text.contains('\u{1b}'), "{text}");
	assert!(!text.contains('\u{7}'), "{text}");
	assert!(text.contains("\nsecond"), "newline layout preserved: {text}");
}

#[cfg(unix)]
#[test]
fn walkers_do_not_follow_symlink_cycles() {
	let dir = TempDir::new("gc-walk-bounds");
	let loop_dir = dir.path().join("sessions/project/loop");
	fs::create_dir_all(&loop_dir).expect("loop dir");
	std::os::unix::fs::symlink(&loop_dir, loop_dir.join("cycle")).expect("cycle symlink");
	dir.write("sessions/project/s.jsonl", "{}");
	assert_eq!(walk_visible_files(&dir.path().join("sessions"), is_jsonl_name).len(), 1);
	assert_eq!(collect_jsonl_files(&loop_dir, ".jsonl").len(), 0);
}

struct ChunkReader {
	data: Vec<u8>,
	pos:  usize,
	step: usize,
}

impl std::io::Read for ChunkReader {
	fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
		let remaining = self.data.len() - self.pos;
		let count = remaining.min(buffer.len()).min(self.step);
		buffer[..count].copy_from_slice(&self.data[self.pos..self.pos + count]);
		self.pos += count;
		Ok(count)
	}
}

#[test]
fn gzip_ref_scan_streams_large_archives_and_chunk_boundaries() {
	let a = "a".repeat(64);
	let b = "b".repeat(64);
	let ref_a = format!("blob:sha256:{a}");
	let ref_b = format!("blob:sha256:{b}");

	let mut payload = vec![b'.'; BLOB_SCAN_CHUNK - 10];
	payload.extend_from_slice(ref_a.as_bytes());
	payload.extend_from_slice(b" tail ");
	payload.extend_from_slice(ref_b.as_bytes());
	let mut hashes = HashSet::new();
	let mut reader = ChunkReader { data: payload, pos: 0, step: 7 };
	scan_blob_refs_reader(&mut reader, Path::new("chunked.jsonl"), &mut hashes).expect("chunked scan");
	assert!(hashes.contains(&a), "reference spanning a chunk boundary");
	assert!(hashes.contains(&b), "reference after the boundary");

	let dir = TempDir::new("gc-gzip-large");
	let gz_path = dir.path().join("large.jsonl.gz");
	let mut encoder = flate2::write::GzEncoder::new(
		fs::File::create(&gz_path).expect("gz"),
		flate2::Compression::fast(),
	);
	use std::io::Write as _;
	let padding = vec![b'x'; DEFAULT_MAX_SCAN_BYTES as usize + 1024 * 1024];
	encoder.write_all(&padding).expect("padding");
	// `/\bblob:/` requires a non-word byte before the reference; padding ends
	// in a word byte, so this newline is part of the fixture's lexical context.
	encoder.write_all(b"\n").expect("separator");
	encoder.write_all(ref_a.as_bytes()).expect("ref");
	encoder.finish().expect("finish");
	let mut hashes = HashSet::new();
	scan_blob_ref_file(&gz_path, true, &mut hashes).expect("large gzip scan");
	assert!(hashes.contains(&a), "large archive refs are not rejected");
}

#[test]
fn gc_agent_dir_uses_its_own_settings() {
	let dir = TempDir::new("gc-agent-settings");
	let agent = dir.path().join("alt-agent");
	dir.write(
		"alt-agent/config.yml",
		"gc:\n  blobs: false\n  archive: false\n  wal: false\n",
	);
	let opts = GcOptions {
		json:                    true,
		agent_dir:               Some(agent),
		blobs:                   false,
		archive:                 false,
		wal:                     false,
		cold_archive_after_days: None,
		retain_newest_global:    None,
		retain_newest_per_cwd:   None,
	};
	let (code, value, stderr) = gc_json(&opts, dir.path());
	assert_eq!(code, 0, "{stderr}");
	assert!(value.get("blobs").is_none(), "{value}");
	assert!(value.get("archive").is_none(), "{value}");
	assert!(value.get("wal").is_none(), "{value}");
}

#[test]
fn gc_agent_dir_text_escapes_controls() {
	let result = json!({ "agentDir": "/tmp/\u{1b}]0;x", "apply": false, "lockPath": "/tmp/x" });
	let text = gc_text(&result);
	assert!(!text.contains('\u{1b}'), "{text}");
	assert!(text.contains("GC dry-run"), "{text}");
}

#[test]
fn ttsr_preview_escapes_terminal_controls() {
	let dir = TempDir::new("ttsr-preview-escape");
	let rule = dir.write(
		"plain.md",
		"---\ncondition: 'trigger'\n---\nBody\n",
	);
	let opts = TtsrOptions {
		action:  Some("test".to_string()),
		snippet: Some("trigger\u{1b}]52;c;x".to_string()),
		file:    None,
		rule:    Some(rule),
		source:  Some("text".to_string()),
		tool:    None,
		path:    None,
		verbose: false,
		json:    false,
		no_gitignore: false,
		max_bytes: None,
	};
	let mut out = Vec::new();
	let mut err = Vec::new();
	run_ttsr(&opts, dir.path(), &mut out, &mut err).expect("ttsr test");
	let text = String::from_utf8(out).expect("utf8");
	assert!(!text.contains('\u{1b}'), "{text}");
	assert!(text.contains("trigger"), "{text}");
}

#[test]
fn python_candidates_use_reported_managed_env_order() {
	let dir = TempDir::new("python-candidates");
	let venv = dir.path().join("project/.venv");
	let venv_python = if cfg!(windows) {
		venv.join("Scripts/python.exe")
	} else {
		venv.join("bin/python")
	};
	dir.write(
		&venv_python.strip_prefix(dir.path()).expect("relative").to_string_lossy(),
		"x",
	);
	let managed = dir.path().join("custom/python-env");
	let managed_python = if cfg!(windows) {
		managed.join("Scripts/python.exe")
	} else {
		managed.join("bin/python")
	};
	dir.write(
		&managed_python.strip_prefix(dir.path()).expect("relative").to_string_lossy(),
		"x",
	);
	let mut base = HashMap::new();
	base.insert("PATH".to_string(), String::new());
	let cwd = dir.path().join("project");
	let candidates = python_candidates(&cwd, &base, None, &managed);
	assert_eq!(
		candidates.first().map(|runtime| runtime.program.as_path()),
		Some(venv_python.as_path())
	);
	assert_eq!(
		candidates.get(1).map(|runtime| runtime.program.as_path()),
		Some(managed_python.as_path())
	);

	let no_venv = dir.path().join("no-venv");
	fs::create_dir_all(&no_venv).expect("cwd");
	let candidates = python_candidates(&no_venv, &base, None, &managed);
	assert_eq!(
		candidates.first().map(|runtime| runtime.program.as_path()),
		Some(managed_python.as_path())
	);
}

#[test]
fn grievances_missing_table_is_empty_but_corrupt_or_busy_fails() {
	let dir = TempDir::new("grievance-errors");
	let opts = GrievancesOptions { limit: 20, tool: None, json: false };

	let missing = dir.path().join("missing-table.db");
	let connection = rusqlite::Connection::open(&missing).expect("db");
	connection
		.execute_batch("CREATE TABLE unrelated (id INTEGER);")
		.expect("schema");
	drop(connection);
	let mut out = Vec::new();
	list_grievances_in(&missing, &opts, &mut out).expect("missing table is empty");
	assert!(String::from_utf8(out).expect("utf8").contains("No grievances recorded yet."));

	let corrupt = dir.path().join("corrupt.db");
	fs::write(&corrupt, b"not a sqlite database").expect("corrupt db");
	let mut out = Vec::new();
	assert!(list_grievances_in(&corrupt, &opts, &mut out).is_err());

	let busy = dir.path().join("busy.db");
	let writer = rusqlite::Connection::open(&busy).expect("busy db");
	writer
		.execute_batch(
			"CREATE TABLE grievances (id INTEGER PRIMARY KEY, model TEXT, version TEXT, tool TEXT, report TEXT);\nPRAGMA locking_mode = EXCLUSIVE;\nBEGIN EXCLUSIVE;",
		)
		.expect("lock");
	let mut out = Vec::new();
	assert!(list_grievances_in(&busy, &opts, &mut out).is_err(), "busy must fail closed");
}

#[test]
fn ttsr_discovers_configured_extension_rules() {
	let dir = TempDir::new("ttsr-extension-rules");
	dir.write(".omp/config.yml", "extensions:\n  - ext\n");
	dir.write(
		"ext/rules/extension-rule.md",
		"---\ncondition: 'extension trigger'\n---\nExtension body\n",
	);
	let rules = discover_rules(dir.path(), &dir.path().join("agent"));
	let rule = rules
		.iter()
		.find(|rule| rule.name == "extension-rule")
		.expect("extension rule");
	assert_eq!(rule.provider, "omp-plugins");
	assert_eq!(rule.condition, Some(vec!["extension trigger".to_string()]));
}
