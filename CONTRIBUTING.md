# Contributing to NVIDIA Trustworthy AI

Thank you for your interest in contributing! This document covers the contribution
rules that apply to **every** component in this repository.

Some components carry additional, component-specific guidance. Where that exists, it
supplements — it does not replace — the requirements on this page:

- [ASL Developer Community/CONTRIBUTING.md](ASL%20Developer%20Community/CONTRIBUTING.md) — coding standards, tests, and workflow for the ASL data pipeline.

## Licensing

This repository contains components under different licenses. Before contributing,
check the license that applies to the files you are changing — the table in the
[README](README.md#licensing) identifies each component's license and links to its
license file. By contributing, you agree that your contribution is licensed under the
license that applies to the component you are modifying.

New source files must carry the SPDX headers required for their component. For
example, an Apache-2.0 source file begins with:

```
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

## Issue Tracking

- Enhancement, bugfix, or change requests should begin with a
  [GitHub issue](https://github.com/NVIDIA/Trustworthy-AI/issues).
- Browse open issues for tasks labeled as good first issues, or open a new issue to
  propose a change.

## Pull Requests

1. [Fork](https://help.github.com/en/articles/fork-a-repo) the repository and create a
   feature branch.
2. Make your changes. Keep each pull request focused on a single concern.
3. Sign off every commit (see [Signing Off Your Work](#signing-off-your-work)).
4. Open a pull request describing what you changed and why, and link any related
   issues (for example, `Closes #18`).
5. At least one project maintainer will review the PR. Mark work in progress by
   prefixing the PR title with `[WIP]` or `[Draft]`.

Write commit titles using the imperative mood and reference the issue number
corresponding to the PR:

```
#<Issue Number> - <Commit Title>

<Commit Body>
```

## Signing Off Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

  ```bash
  $ git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO (https://developercertificate.org/):

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
  ```

## Questions

For questions or suggestions, open a
[GitHub issue](https://github.com/NVIDIA/Trustworthy-AI/issues).
