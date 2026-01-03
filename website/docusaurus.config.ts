import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'kiss3d',
  tagline: 'Keep It Simple, Stupid - 3D and 2D graphics for Rust',
  favicon: 'img/kiss3d-logo-small.png',

  future: {
    v4: true,
  },

  url: 'https://kiss3d.rs',
  baseUrl: '/',

  organizationName: 'sebcrozet',
  projectName: 'kiss3d',

  onBrokenLinks: 'throw',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: false,
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/kiss3d-social-card.png',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'kiss3d',
      logo: {
        alt: 'kiss3d Logo',
        src: 'img/kiss3d-logo-small.png',
      },
      items: [
        {
          to: '/examples',
          label: 'Examples',
          position: 'left',
        },
        {
          href: 'https://docs.rs/kiss3d',
          label: 'API Docs',
          position: 'left',
        },
        {
          href: 'https://github.com/sebcrozet/kiss3d',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Resources',
          items: [
            {
              label: 'API Documentation',
              href: 'https://docs.rs/kiss3d',
            },
            {
              label: 'Examples',
              href: 'https://github.com/sebcrozet/kiss3d/tree/master/examples',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub Discussions',
              href: 'https://github.com/sebcrozet/kiss3d/discussions',
            },
            {
              label: 'Issues',
              href: 'https://github.com/sebcrozet/kiss3d/issues',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/sebcrozet/kiss3d',
            },
            {
              label: 'Crates.io',
              href: 'https://crates.io/crates/kiss3d',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} kiss3d contributors. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['rust', 'toml', 'bash'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
