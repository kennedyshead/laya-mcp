"""Write a balanced, privacy-safe reviewed MCP-routing seed corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TASKS = {
    "jcodemunch": [
        "Locate the definition of the retry helper.",
        "Find every caller of the authentication function.",
        "Show the symbol outline for the API module.",
        "Trace the call hierarchy for the request handler.",
        "Find implementations of the storage interface.",
        "Calculate the blast radius of changing the parser.",
        "Show where the timeout constant is referenced.",
        "Find tests covering the serializer.",
        "Locate code that handles connection errors.",
        "Find the repository entry point.",
        "Show the source for the cache invalidation method.",
        "Identify files importing the configuration module.",
        "Find code that performs token validation.",
        "Locate the database transaction helpers.",
        "Find all subclasses of the base command class.",
        "Show the changed symbols between two revisions.",
        "Find callers of the HTTP client wrapper.",
        "Locate logging calls in the worker module.",
        "Find the code path for request retries.",
        "Show all references to the feature flag.",
    ],
    "docs-mcp-server": [
        "Look up the current API for the HTTP client library.",
        "Find installation instructions for the framework.",
        "Check supported versions of the database driver.",
        "Find migration guidance for the dependency upgrade.",
        "Look up the framework's authentication documentation.",
        "Find the API reference for the configuration class.",
        "Check the library's timeout configuration options.",
        "Find an official example of file uploads.",
        "Look up middleware documentation for the framework.",
        "Check deployment instructions for the package.",
        "Find the caching configuration reference.",
        "Look up the library's rate-limit guidance.",
        "Find webhook verification documentation.",
        "Check the framework lifecycle hook reference.",
        "Find the documented background-job configuration.",
        "Look up the package environment variables.",
        "Find pagination examples in the current API docs.",
        "Check the library's security recommendations.",
        "Look up the deprecation notice for this API.",
        "Find official testing guidance for the framework.",
    ],
    "gitea": [
        "List the open issues in this repository.",
        "Find issues matching a keyword.",
        "Show the details of an issue.",
        "Show the comments on an issue.",
        "Create a repository issue.",
        "Update an issue title.",
        "Add a comment to an issue.",
        "Close an issue.",
        "Reopen an issue.",
        "Add a label to an issue.",
        "Assign an issue to a contributor.",
        "List the repository branches.",
        "List open pull requests.",
        "Show pull request details.",
        "Add a pull request review comment.",
        "List recent commits.",
        "Create a release from a tag.",
        "Show the latest release.",
        "Create a branch from main.",
        "Find repositories by name.",
    ],
    None: [
        "Explain this error message.",
        "Suggest names for this function.",
        "Rewrite this sentence concisely.",
        "Summarize the supplied text.",
        "Convert this list to a table.",
        "Draft a neutral status update.",
        "Generate test-case ideas.",
        "Explain this programming concept.",
        "Compare these two supplied approaches.",
        "Extract action items from these notes.",
        "Produce a checklist from these requirements.",
        "Reformat this JSON.",
        "Draft release-note wording.",
        "Improve this commit-message draft.",
        "Create a meeting agenda.",
        "Generate edge cases for this input.",
        "Draft documentation headings.",
        "Turn these notes into bullets.",
        "Create placeholder example data.",
        "Explain this regular expression.",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    rows = [
        {"task": task, "label": label, "label_status": "reviewed"}
        for label, tasks in TASKS.items()
        for task in tasks
    ]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8") as corpus:
        for row in rows:
            corpus.write(json.dumps(row, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
