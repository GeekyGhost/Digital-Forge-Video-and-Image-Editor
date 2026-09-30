# Contributing

Thanks for your interest in Digital Forge Video Studio.

## Licensing of contributions

By submitting a pull request, issue attachment, or any other contribution, you
agree that your contribution is licensed under the same terms as this project —
the **Functional Source License 1.1 with an Apache 2.0 future licence**
(`FSL-1.1-ALv2`), as set out in [LICENSE.md](LICENSE.md).

You also confirm that you have the right to submit the work under those terms,
and that it is your own or is compatibly licensed and clearly attributed.

## Before opening a pull request

- Run `npm.cmd run check` for the TypeScript build.
- Run `Test-AIV-Studio.bat` for the qualification suite when you touch export,
  timeline, or adapter code.
- Python worker changes should keep `python -m pytest ai-runtime/tests` green.

## Scope notes

- **Do not commit model weights, `vendor/` sources, or anything under
  `projects/`.** These are excluded in `.gitignore` for good reasons — size and
  privacy respectively.
- Adapters live behind the JSON-lines worker protocol in `ai-runtime/`. Adding a
  model means adding a manifest in `models/manifests/` and an adapter, not
  reaching into the editor core.
