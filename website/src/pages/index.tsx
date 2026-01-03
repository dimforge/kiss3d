import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import CodeBlock from '@theme/CodeBlock';

import styles from './index.module.css';

const cargoToml = `[package]
name = "my-kiss3d-app"
version = "0.1.0"
edition = "2024"

[dependencies]
kiss3d = "0.38"`;

const cubeExample = `use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: cube").await;
    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 2.0, -2.0));

    let mut c = scene.add_cube(1.0, 1.0, 1.0).set_color(RED);

    let rot = Quat::from_axis_angle(Vec3::Y, 0.014);

    while window.render_3d(&mut scene, &mut camera).await {
        c.rotate(rot);
    }
}`;

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <img
          src="/img/kiss3d-logo.png"
          alt="kiss3d logo"
          className={styles.heroLogo}
        />
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/examples">
            View Examples
          </Link>
          <Link
            className="button button--outline button--secondary button--lg"
            to="https://github.com/sebcrozet/kiss3d">
            GitHub
          </Link>
        </div>
      </div>
    </header>
  );
}

function CodeExample(): ReactNode {
  return (
    <section className={styles.codeSection}>
      <div className="container">
        <div className="row">
          <div className="col col--5">
            <Heading as="h2">Simple by Design</Heading>
            <p>
              Kiss3d is simple 3D and 2D graphics engine in Rust based on
              WebGpu. It aims to Keep It Simple Stupid for rendering simple
              geometric primitives and making small interactive demos.
            </p>
            <p>
              Cross-platform support for Windows, macOS, Linux, and WebAssembly.
            </p>
            <div className={styles.codeLinks}>
              <Link
                className="button button--primary"
                to="/examples">
                Browse Examples
              </Link>
              <Link
                className="button button--outline button--primary"
                to="https://docs.rs/kiss3d">
                API Documentation
              </Link>
            </div>
          </div>
          <div className="col col--7">
            <CodeBlock language="rust" title="src/main.rs">
              {cubeExample}
            </CodeBlock>
            <CodeBlock language="toml" title="Cargo.toml">
              {cargoToml}
            </CodeBlock>
          </div>
        </div>
      </div>
    </section>
  );
}

function Features(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.featureGrid}>
          <div className={styles.feature}>
            <span className={styles.featureIcon}>🎯</span>
            <h3>Simple API</h3>
            <p>One-liner for most operations. Add objects, set colors, render — done.</p>
          </div>
          <div className={styles.feature}>
            <span className={styles.featureIcon}>🌐</span>
            <h3>Cross-Platform</h3>
            <p>Windows, macOS, Linux, and WebAssembly from the same codebase.</p>
          </div>
          <div className={styles.feature}>
            <span className={styles.featureIcon}>🎨</span>
            <h3>Modern Graphics</h3>
            <p>Powered by wgpu — Vulkan, DirectX 12, Metal, and WebGPU.</p>
          </div>
          <div className={styles.feature}>
            <span className={styles.featureIcon}>📦</span>
            <h3>3D & 2D</h3>
            <p>Cubes, spheres, rectangles, circles, lines, and custom meshes.</p>
          </div>
          <div className={styles.feature}>
            <span className={styles.featureIcon}>💡</span>
            <h3>Lighting</h3>
            <p>Point, directional, and spot lights with configurable properties.</p>
          </div>
          <div className={styles.feature}>
            <span className={styles.featureIcon}>📷</span>
            <h3>Cameras</h3>
            <p>Orbit, first-person, and stereo cameras ready to use.</p>
          </div>
        </div>
      </div>
    </section>
  );
}

function Install(): ReactNode {
  return (
    <section className={styles.install}>
      <div className="container">
        <Heading as="h2">Get Started</Heading>
        <div className={styles.installBox}>
          <code>cargo add kiss3d</code>
        </div>
        <p className={styles.installNote}>
          Then check out the <Link to="/examples">examples</Link> or read the{' '}
          <Link to="https://docs.rs/kiss3d">API documentation</Link>.
        </p>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  return (
    <Layout
      title="Simple 3D and 2D Graphics for Rust"
      description="Kiss3d is a simple, cross-platform 3D and 2D graphics engine for Rust. Perfect for visualizations, simulations, and learning graphics programming.">
      <HomepageHeader />
      <main>
        <CodeExample />
      </main>
    </Layout>
  );
}
