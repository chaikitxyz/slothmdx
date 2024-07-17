# SlothMdx Astro

This is a [Astro](https://astro.build) project, inspired from [Astro Cactus](https://astro.build/themes/details/astro-cactus) and [Astropod](https://astro.build/themes/details/astropod-free-serverless-podcast)

## Project Structure

Inside of your Astro project, you'll see the following folders and files structure:

```text
/
├── public/
│   └── favicon.svg
├── src/
│   ├── components/
│   │   └── Search.astro
│   ├── content/
│   │   ├── post/
│   │   ├── podcast/
│   │   └── config.ts
│   ├── layouts/
│   │   └── Layout.astro
│   └── pages/
│       └── index.astro
└── package.json
```

Astro looks for `.astro` or `.md` files in the `src/pages/` directory. Each page is exposed as a route based on its file name.

There's nothing special about `src/components/`, but that's where we like to put any Astro/React/Vue/Svelte/Preact components.

Any static assets, like images, can be placed in the `public/` directory.

## Getting Started

You can change the site's base configuration in 
`src/site.config.ts`. This is where you can set the site's title, description, and other metadata.
Also can change links for both header and footer with the `menuLinks`.

Theme options: `THEME.md` file

## Commands

All commands are run from the root of the project, from a terminal:

| Command                   | Action                                           |
| :------------------------ | :----------------------------------------------- |
| `npm install`             | Installs dependencies                            |
| `npm run dev`             | Starts local dev server at `localhost:4321`      |
| `npm run build`           | Build your production site to `./dist/`          |
| `npm run preview`         | Preview your build locally, before deploying     |
| `npm run astro ...`       | Run CLI commands like `astro add`, `astro check` |
| `npm run astro -- --help` | Get help using the Astro CLI                     |
