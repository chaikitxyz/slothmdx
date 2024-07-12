import type { SiteConfig } from "@/types";

export const siteConfig: SiteConfig = {
	// Used as both a meta property (src/components/BaseHead.astro L:31 + L:49) & the generated satori png (src/pages/og-image/[slug].png.ts)
	author: "Montek Kundan",
	// Meta property used to construct the meta title property, found in src/components/BaseHead.astro L:11
	title: "Montek Devlog",
	// Meta property used as the default description meta property
	description: "My digital garden of Logs + Projects",
	// HTML lang property, found in src/layouts/Base.astro L:18
	lang: "en",
	// Meta property, found in src/components/BaseHead.astro L:42
	ogLocale: "en",
	// Date.prototype.toLocaleDateString() parameters, found in src/utils/date.ts.
	date: {
		locale: "en",
		options: {
			day: "numeric",
			month: "short",
			year: "numeric",
		},
	},
	header: {
		logo : {
			url: "https://montek.dev",
			name: "Montek"
		},
		blogTitle: "Devlog"
	},
	index: {
		title: "Hello World!",
		subtitle: "Hi! My digital garden contains all my logs about the projects I find interesting. I hope you learn something from my logs and projects!",
	},
	about: {
		title: "About",
		subtitle: `Hi, you can learn more about me on montek.dev`,
	},
	// Theme options: "Sunset Forest", "Autumn Grove", "Sunrise Orchid", "Crimson Tide"
	theme: "Sunset Forest"
};

// Used to generate links in both the Header & Footer.
export const menuLinks: Array<{ title: string; path: string }> = [
	{
		title: "Home",
		path: "/",
	},
	{
		title: "About",
		path: "/about/",
	},
	{
		title: "Devlog",
		path: "/posts/",
	},
	{
		title: "Podcast",
		path: "/podcast/",
	},
];

export const POSTS_PER_PAGE = 5;
export const PODCASTS_PER_PAGE = 2;
