# Configuration Fields

## Basic Information

- `author`: The name of the author of the site. Used as a meta property.
- `title`: The title of the site. Used to construct the meta title property.
- `description`: A brief description of the site. Used as the default description meta property.
- `lang`: The language of the site content. Set as the HTML `lang` attribute.
- `ogLocale`: The Open Graph locale property.

## Date Formatting

- `date`: An object containing locale and formatting options for dates.
  - `locale`: The locale for date formatting (e.g., "en" for English).
  - `options`: Formatting options for `Date.prototype.toLocaleDateString()`.

## Header Configuration

- `header`: Configuration for the site's header.
  - `logo`: An object with logo details.
    - `url`: The URL of the logo.
    - `name`: The name of the logo.
  - `blogTitle`: The title of the blog shown in the header.

## Index Page

- `index`: Configuration for the index (home) page.
  - `title`: The main title of the index page.
  - `subtitle`: A subtitle for the index page, providing additional context.

## Social Links

- `socialLinks`: Configuration for social links section.
  - `title`: The title of the social links section.
- `socialLinksList`: An array of social link objects.
  - `name`: The icon name for the social platform.
  - `friendlyName`: The display name of the social platform.
  - `link`: The URL to the social platform profile.

## About Page

- `about`: Configuration for the about page.
  - `title`: The title of the about page.
  - `subtitle`: A subtitle for the about page, usually a brief introduction about the author.

## Podcast Page

- `podcast`: Configuration for the podcast page.
  - `title`: The title of the podcast page.
  - `subtitle`: A subtitle for the podcast page, providing context about the content.

## Theme

- `theme`: The theme of the site. Options include "Sunset Forest", "Autumn Grove", "Sunrise Orchid", and "Crimson Tide".

## Menu Links

- `menuLinks`: An array of objects representing the menu links displayed in the header and footer.
  - `title`: The title of the menu item.
  - `path`: The path to the menu item.

## Pagination

- `POSTS_PER_PAGE`: The maximum number of posts to display on a single page.
- `PODCASTS_PER_PAGE`: The maximum number of podcasts to display on a single page.
