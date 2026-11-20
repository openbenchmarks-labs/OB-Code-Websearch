# tools

One `<tool>.yaml` file per search tool. Keeping each tool in its own file makes it easy to compare setups and open pull requests without merge conflicts.

Each file lists the tool's setups (`main`, `settings`, or `comparison`) and their pricing: `list_usd`, the list price per search request, and, for setups that also fetch pages, `fetch_usd` per page with its public source. The run's spend cap counts `list_usd` and `fetch_usd`.

To add or update a tool: edit `tools/<tool>.yaml`, run the offline `selftest`, and open a pull request.
