//! Read-only `security://` resource namespace.
//!
//! Anchors: `packages/coding-agent/src/internal-urls/security-protocol.ts`.
//! The handler resolves the project store for the read context's cwd (git root
//! when inside a repository) and exposes scans, findings, coverage, report,
//! SARIF, and redacted provenance. It never mutates the store and reports the
//! exact disabled message when `security.enabled` is false.

use std::path::PathBuf;

use async_trait::async_trait;

use pi_agent::tools::internal_urls::{
	InternalResource, InternalUrl, ProtocolHandler, ResolveContext,
};

use super::{
	provenance::{create_public_security_scan, redact_private_security_metadata},
	store::{SecurityScanSummary, SecurityStore},
};

pub const SECURITY_SCHEME: &str = "security";
pub const SECURITY_DISABLED_MESSAGE: &str =
	"security:// is disabled. Enable it by setting `security.enabled = true` (Settings → Tools → Security).";
const MAX_BYTES: usize = 50 * 1024;
const MAX_LINES: usize = 3000;

/// Mutable-free handler mounted on a session's internal URL router.
#[derive(Debug, Clone)]
pub struct SecurityUrlHandler {
	enabled: bool,
	state_root: Option<PathBuf>,
}

impl SecurityUrlHandler {
	#[must_use]
	pub fn new(enabled: bool, state_root: Option<PathBuf>) -> Self {
		Self { enabled, state_root }
	}

	fn store(&self, context: &ResolveContext) -> Result<SecurityStore, String> {
		// `SecurityStore::open` resolves the enclosing worktree itself, so the
		// namespace sees the same project whether the read runs at the root or in
		// a subdirectory.
		SecurityStore::open(&context.cwd, self.state_root.as_deref())
			.map_err(|error| error.to_string())
	}

	fn split(url: &InternalUrl) -> Vec<String> {
		let mut parts: Vec<String> = Vec::new();
		if !url.host.is_empty() {
			parts.push(url.host.clone());
		}
		for segment in url.path.split('/') {
			if !segment.is_empty() {
				parts.push(segment.to_string());
			}
		}
		parts
	}
}

#[async_trait]
impl ProtocolHandler for SecurityUrlHandler {
	fn scheme(&self) -> &str {
		SECURITY_SCHEME
	}

	fn immutable(&self) -> bool {
		true
	}

	fn writable(&self) -> bool {
		false
	}

	async fn resolve(
		&self,
		url: &InternalUrl,
		context: &ResolveContext,
	) -> Result<InternalResource, String> {
		if !self.enabled {
			return Err(SECURITY_DISABLED_MESSAGE.to_string());
		}
		let parts = Self::split(url);
		let store = self.store(context)?;
		if parts.is_empty() {
			return Ok(bounded_resource(
				"security://",
				[
					"# Security",
					"",
					"OMP-owned software-security analysis resources. The namespace is read-only; use explicit security commands or tools for mutations.",
					"",
					"- `security://scans` — list scans",
					"",
				]
				.join("\n"),
				"text/markdown",
				true,
			));
		}
		if parts[0] != "scans" {
			return Err(format!("Unknown security resource: security://{}", parts.join("/")));
		}
		if parts.len() == 1 {
			return Ok(bounded_resource(
				"security://scans",
				format_scans(&store.list_scans().map_err(|error| error.to_string())?),
				"text/markdown",
				true,
			));
		}
		let scan_id = &parts[1];
		let bundle = store
			.get_bundle(scan_id)
			.map_err(|error| error.to_string())?
			.ok_or_else(|| format!("Unknown security scan: {scan_id}"))?;
		if parts.len() == 2 {
			return Ok(bounded_resource(
				&format!("security://scans/{scan_id}"),
				[
					format!("# Security scan {scan_id}"),
					String::new(),
					format!("- Status: **{}**", bundle.scan.status),
					format!("- Producer: **{}**", sanitize(&bundle.scan.producer.name)),
					format!("- Findings: **{}**", bundle.findings.len()),
					format!("- Coverage: **{}**", bundle.scan.coverage.completeness),
					format!("- Target: `{}`", sanitize(&bundle.scan.target.display_name)),
					String::new(),
					"Resources: `manifest`, `findings`, `coverage`, `report`, `sarif`, `provenance`.".to_string(),
					String::new(),
				]
				.join("\n"),
				"text/markdown",
				true,
			));
		}
		match parts[2].as_str() {
			"manifest" => {
				if parts.len() != 3 {
					return Err(format!("Unknown security resource: security://{}", parts.join("/")));
				}
				let public = create_public_security_scan(&bundle.scan, true)
					.map_err(|error| error.to_string())?;
				Ok(bounded_resource(
					&format!("security://scans/{scan_id}/manifest"),
					format!("{}\n", serde_json::to_string_pretty(&public).unwrap_or_default()),
					"application/json",
					false,
				))
			},
			"findings" => {
				if parts.len() == 3 {
					let listing: Vec<String> = bundle
						.findings
						.iter()
						.map(|finding| {
							format!(
								"- `{}` **{}** — {} (`{}`)",
								finding.id,
								finding.severity.level,
								sanitize(&finding.title),
								sanitize(&finding.rule_id)
							)
						})
						.collect();
					return Ok(bounded_resource(
						&format!("security://scans/{scan_id}/findings"),
						format!(
							"# Findings for {scan_id}\n\n{}\n",
							if listing.is_empty() { "No findings.".to_string() } else { listing.join("\n") }
						),
						"text/markdown",
						true,
					));
				}
				if parts.len() != 4 {
					return Err(format!("Unknown security resource: security://{}", parts.join("/")));
				}
				let finding_id = &parts[3];
				let finding = bundle
					.finding(finding_id)
					.ok_or_else(|| format!("Unknown security finding: {finding_id}"))?;
				Ok(bounded_resource(
					&format!("security://scans/{scan_id}/findings/{finding_id}"),
					format_finding(finding),
					"text/markdown",
					false,
				))
			},
			"coverage" => {
				if parts.len() != 3 {
					return Err(format!("Unknown security resource: security://{}", parts.join("/")));
				}
				Ok(bounded_resource(
					&format!("security://scans/{scan_id}/coverage"),
					format!(
						"{}\n",
						serde_json::to_string_pretty(&bundle.scan.coverage).unwrap_or_default()
					),
					"application/json",
					false,
				))
			},
			"report" => {
				if parts.len() != 3 {
					return Err(format!("Unknown security resource: security://{}", parts.join("/")));
				}
				let report = bundle
					.report
					.as_deref()
					.ok_or_else(|| format!("Security scan {scan_id} has no report"))?;
				Ok(bounded_resource(
					&format!("security://scans/{scan_id}/report"),
					report.to_string(),
					"text/markdown",
					false,
				))
			},
			"sarif" => {
				if parts.len() != 3 {
					return Err(format!("Unknown security resource: security://{}", parts.join("/")));
				}
				let sarif = bundle
					.sarif
					.as_ref()
					.ok_or_else(|| format!("Security scan {scan_id} has no SARIF export"))?;
				Ok(bounded_resource(
					&format!("security://scans/{scan_id}/sarif"),
					format!("{}\n", serde_json::to_string_pretty(sarif).unwrap_or_default()),
					"application/json",
					false,
				))
			},
			"provenance" => {
				if parts.len() != 3 {
					return Err(format!("Unknown security resource: security://{}", parts.join("/")));
				}
				let value = serde_json::to_value(&bundle.scan.provenance)
					.map_err(|error| error.to_string())?;
				let redacted = redact_private_security_metadata(&value);
				Ok(bounded_resource(
					&format!("security://scans/{scan_id}/provenance"),
					format!("{}\n", serde_json::to_string_pretty(&redacted).unwrap_or_default()),
					"application/json",
					false,
				))
			},
			_ => Err(format!("Unknown security resource: security://{}", parts.join("/"))),
		}
	}
}

fn format_scans(scans: &[SecurityScanSummary]) -> String {
	if scans.is_empty() {
		return "# Security scans\n\nNo scans are stored for this project.\n".to_string();
	}
	let rows: Vec<String> = scans
		.iter()
		.map(|scan| {
			format!(
				"- `{}` — {}; {} finding(s); {}; {}",
				scan.id,
				scan.status,
				scan.finding_count,
				scan.producer.name,
				scan.created_at
			)
		})
		.collect();
	format!("# Security scans\n\n{}\n", rows.join("\n"))
}

fn format_finding(finding: &super::types::SecurityFinding) -> String {
	let locations: Vec<String> = finding
		.occurrences
		.iter()
		.flat_map(|occurrence| occurrence.locations.iter())
		.map(|location| {
			let end = location
				.end_line
				.filter(|end| Some(*end) != Some(location.start_line))
				.map(|end| format!("-{end}"))
				.unwrap_or_default();
			format!(
				"- `{}:{}{}`{}",
				sanitize(&location.path),
				location.start_line,
				end,
				location
					.role
					.as_ref()
					.map(|role| format!(" ({})", sanitize(role)))
					.unwrap_or_default()
			)
		})
		.collect();
	let evidence: Vec<String> = finding
		.evidence
		.iter()
		.map(|item| format!("- **{}** — {}", sanitize(&item.label), sanitize(&item.explanation)))
		.collect();
	[
		format!("# {}", sanitize(&finding.title)),
		String::new(),
		format!("- ID: `{}`", finding.id),
		format!("- Rule: `{}`", sanitize(&finding.rule_id)),
		format!("- Severity: **{}**", finding.severity.level),
		format!("- Confidence: **{}**", finding.confidence.level),
		format!("- Disposition: **{}**", finding.disposition.status),
		format!("- Fingerprint: `{}`", finding.fingerprint),
		String::new(),
		"## Summary".to_string(),
		String::new(),
		sanitize(&finding.summary),
		String::new(),
		"## Locations".to_string(),
		String::new(),
		if locations.is_empty() {
			"No source locations recorded.".to_string()
		} else {
			locations.join("\n")
		},
		String::new(),
		"## Evidence".to_string(),
		String::new(),
		if evidence.is_empty() {
			"No expanded evidence recorded.".to_string()
		} else {
			evidence.join("\n")
		},
		String::new(),
		"## Remediation".to_string(),
		String::new(),
		sanitize(
			finding
				.remediation
				.as_deref()
				.unwrap_or("No remediation guidance recorded."),
		),
		String::new(),
	]
	.join("\n")
}

fn sanitize(text: &str) -> String {
	text.chars().filter(|ch| *ch != '\u{1b}' && !ch.is_control()).collect()
}

fn bounded_resource(url: &str, content: String, content_type: &str, is_directory: bool) -> InternalResource {
	let sanitized = sanitize(&content);
	let lines: Vec<&str> = sanitized.lines().collect();
	let mut bounded = String::new();
	let mut truncated = false;
	let mut line_count = 0;
	for line in lines {
		if line_count >= MAX_LINES || bounded.len() + line.len() + 1 > MAX_BYTES {
			truncated = true;
			break;
		}
		bounded.push_str(line);
		bounded.push('\n');
		line_count += 1;
	}
	if content_type == "application/json" && truncated {
		let original_lines = sanitized.lines().count();
		bounded = format!(
			"{}\n",
			serde_json::to_string_pretty(&serde_json::json!({
				"truncated": true,
				"originalBytes": sanitized.len(),
				"originalLines": original_lines,
				"preview": bounded,
			}))
			.unwrap_or_default()
		);
	}
	InternalResource {
		url: url.to_string(),
		size: bounded.len(),
		content: bounded,
		content_type: content_type.to_string(),
		source_path: None,
		notes: truncated.then(|| {
			vec![format!(
				"Security resource truncated to {MAX_LINES} lines / {MAX_BYTES} bytes."
			)]
		}),
		immutable: true,
		is_directory,
		writable: false,
	}
}
