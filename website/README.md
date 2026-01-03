# kiss3d Website

This website is built using [Docusaurus](https://docusaurus.io/) and includes interactive WebAssembly demos of kiss3d examples.

## Prerequisites

- [Node.js](https://nodejs.org/) (v18 or later)
- [Rust](https://rustup.rs/) with the `wasm32-unknown-unknown` target
- [wasm-bindgen-cli](https://rustwasm.github.io/wasm-bindgen/)

Install the required Rust tooling:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli
```

## Installation

```bash
npm install
```

## Building the Demos

The website includes interactive demos compiled from kiss3d examples to WebAssembly.

Build all demos:

```bash
npm run build:demos
```

Build a single demo:

```bash
npm run build:demo <example_name>
# e.g., npm run build:demo cube
```

The demos are built to `static/demos/` and will be included in the website.

## Local Development

```bash
npm start
```

This starts a local development server at http://localhost:3000. Most changes are reflected live without restarting the server.

## Build for Production

Build everything (demos + website):

```bash
npm run build:all
```

Or build just the website (assumes demos are already built):

```bash
npm run build
```

The static site is generated in the `build` directory.

## Deployment

The website can be deployed to any static hosting service. For GitHub Pages:

```bash
GIT_USER=<Your GitHub username> npm run deploy
```

## Project Structure

```
website/
├── src/
│   ├── pages/          # React pages (index, examples)
│   └── css/            # Custom styles
├── static/
│   ├── demos/          # Compiled WASM demos
│   └── img/            # Images and logos
├── scripts/
│   └── build-demos.sh  # Demo build script
└── docusaurus.config.ts
```
