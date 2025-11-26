# Contributing to ASL Data Pipeline

First off, thank you for considering contributing to ASL Data Pipeline! It's people like you that make open source such a great community.

## Contribution Rules

### Issue Tracking

- All enhancement, bugfix, or change requests must begin with the creation of an [Issue](https://github.com/NVIDIA/asl-data-pipeline/issues).
- The issue must be reviewed by the project maintainers and approved prior to code review.

### Coding Guidelines

While we don't have a strict coding guideline document, we encourage contributions that align with the existing code style.

- Please follow the existing conventions in the relevant file, submodule, module, and project when you add new code or when you extend/fix existing functionality.
- To maintain consistency in code formatting and style, you should run a formatter on the modified sources. We recommend using `black`.

  ```bash
  # Install black if you haven't already
  pip install black

  # Format your changed files
  black .
  ```

- Avoid introducing unnecessary complexity into existing code so that maintainability and readability are preserved.
- Try to keep pull requests (PRs) as concise as possible:
  - Avoid committing commented-out code.
  - Wherever possible, each PR should address a single concern. If there are several otherwise-unrelated things that should be fixed to reach a desired endpoint, our recommendation is to open several PRs and indicate the dependencies in the description.
- Write commit titles using imperative mood and [these rules](https://chris.beams.io/posts/git-commit/), and reference the Issue number corresponding to the PR. Following is the recommended format for commit texts:

  ```
  #<Issue Number> - <Commit Title>

  <Commit Body>
  ```

- Ensure that your code is free of linting errors.
- Ensure that all tests pass prior to submitting your code. You can run tests using `pytest`.
- All new components must have an accompanying test.
- All new components must contain accompanying documentation (in code or in READMEs) describing the functionality.
- Make sure that you can contribute your work to open source (no license and/or patent conflict is introduced by your code). You will need to [`sign`](#signing-your-work) your commit.
- Thanks in advance for your patience as we review your contributions; we do appreciate them!

### Pull Requests

Our developer workflow for code contributions is as follows:

1.  Developers must first [fork](https://help.github.com/en/articles/fork-a-repo) the [upstream](https://github.com/NVIDIA/asl-data-pipeline) ASL Data Pipeline repository.
2.  Git clone the forked repository and push changes to the personal fork.
    ```bash
    git clone https://github.com/YOUR_USERNAME/asl-data-pipeline.git
    # Checkout the targeted branch and commit changes
    # Push the commits to a branch on the fork (remote).
    git push -u origin <local-branch>:<remote-branch>
    ```
3.  Once the code changes are staged on the fork and ready for review, a [Pull Request](https://help.github.com/en/articles/about-pull-requests) (PR) can be [requested](https://help.github.com/en/articles/creating-a-pull-request) to merge the changes from a branch of the fork into a selected branch of upstream.
    - Exercise caution when selecting the source and target branches for the PR.
    - Creation of a PR kicks off the code review process.
    - At least one project maintainer will be assigned for the review.
    - While under review, mark your PRs as a draft or as work-in-progress by prefixing the PR title with `[WIP]` or `[Draft]`.
4.  The PR will be accepted and the corresponding issue closed only after the code has been reviewed and any required tests are passing.

### Signing Your Work

We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license. Any contribution which contains commits that are not Signed-Off will not be accepted.

To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

```bash
$ git commit -s -m "Add cool feature."
```

This will append the following to your commit message:

```
Signed-off-by: Your Name <your@email.com>
```

Full text of the DCO:

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.
```

```
Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the
    best of my knowledge, is covered under an appropriate open source
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
    personal information I submit with it, including my sign-off)
    is maintained indefinitely and may be redistributed consistent
    with this project or the open source license(s) involved.
```
