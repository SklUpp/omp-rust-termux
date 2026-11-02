//! Plan-yolo CLI surface: the flags are parsed as options, never as prompt
//! text, and `--plan-yolo-into` is retained for the startup hard-error check.

use std::ffi::OsString;

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
