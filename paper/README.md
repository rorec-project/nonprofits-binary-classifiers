# Papers

This directory intentionally holds two different writing surfaces.

- `techreport/` is the tracked, self-contained computational social-science technical report for this repository. It documents the classifier's methods, findings, and evidence from beginning to end.
- `draft/` is an ignored local symlink to the shared Overleaf project. That project synthesizes results from this and sister repositories for an economics audience. It is not a versioned artifact of this repository.

Create the local draft view with:

```bash
ln -s "/home/dubidub/Cloud/Sync/Dropbox/Apps/Overleaf/Nonprofit summary" paper/draft
```

Copy figures into the Overleaf project when needed. Do not add relative links from the shared draft back to artifacts in this repository; the Overleaf project must remain independently buildable.
