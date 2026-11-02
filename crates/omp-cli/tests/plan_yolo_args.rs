//! Plan-yolo CLI surface: the flags are parsed as options, never as prompt
//! text, and `--plan-yolo-into` is retained for the startup hard-error check.

use std::{ffi::OsString, process::Command};

use clap::Parser;
use omp_cli::args::{Cli, parse_cli};

fn parse(args: &[&str]) -> Cli {
	parse_cli(args.iter().map(OsString::from).collect())
}

#[test]
fn plan_yolo_flags_parse_before_the_prompt() {
	let cli = parse(&[
		"omp",
		"--plan-yolo",
		"--plan-yolo-into",
		"@slow",
		"implement the plan",
	]);
	assert!(cli.plan_yolo);
	assert_eq!(cli.plan_yolo_into.as_deref(), Some("@slow"));
	assert_eq!(cli.prompt, vec!["implement the plan".to_string()]);
}

#[test]
fn plan_yolo_defaults_to_no_target_override() {
	let cli = parse(&["omp", "--plan-yolo", "hello"]);
	assert!(cli.plan_yolo);
	assert_eq!(cli.plan_yolo_into, None);
	assert_eq!(cli.prompt, vec!["hello".to_string()]);
}

#[test]
fn explicit_prewalk_and_plan_yolo_conflict_before_startup() {
	let error = Cli::try_parse_from([
		"omp",
		"--prewalk",
		"--plan-yolo",
		"prompt",
		"must not run",
	])
	.expect_err("both one-way startup switches must be rejected");
	assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);
}

#[test]
fn explicit_prewalk_and_plan_yolo_exit_before_contact() {
	let output = Command::new(env!("CARGO_BIN_EXE_omp"))
		.args([
			"--prewalk-into",
			"@smol",
			"--plan-yolo-into",
			"@slow",
			"must not run",
		])
		.output()
		.expect("run binary");
	assert_eq!(output.status.code(), Some(2), "argument conflict is a usage exit");
	let stderr = String::from_utf8_lossy(&output.stderr);
	assert!(
		stderr.contains("cannot be used with") || stderr.contains("conflict"),
		"stderr={stderr}"
	);
	assert!(output.stdout.is_empty(), "nothing may start before the conflict");
}
