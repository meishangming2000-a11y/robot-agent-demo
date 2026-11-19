# Release validation

Validation results for this curated source snapshot are recorded in
`evidence/release-validation.json` before publication. Source code and the
project skill were copied byte-for-byte from the active research project;
`SOURCE_SNAPSHOT.json` records their SHA256 hashes. Release-specific documentation,
CI and ignore rules are authored for this distribution.

Checks cover the packaged automated tests, actual nominal reference episodes,
independent physical gate recomputation, local Markdown links, and a scan of
publishable files for credentials and private workspace paths. Robot assets are
pinned and hash-checked; downloaded assets and generated runs are excluded from Git.
Historical experiment results are separately attributed in [results](results.md).

CI runs the 1,001 non-rendering checks on Windows, including the historical
Windows XML byte hashes and Windows file-sharing semantics. A separate Ubuntu
job runs the same real rendered smoke test with `MUJOCO_GL=osmesa` and
`libosmesa6`. This is MuJoCo's supported
[headless software-rendering route](https://mujoco.readthedocs.io/en/stable/programming/rendering.html).
The first hosted Windows run passed 1,001 tests but failed that render test
because its graphics driver did not support OpenGL. Local Windows rendering
and the three complete reference tasks passed. A subsequent all-Linux run
passed rendering but did not reproduce four historical Windows XML byte hashes.
The split jobs cover every packaged test without changing their assertions or
the runtime. No rendering test or physical acceptance condition was removed.
