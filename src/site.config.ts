import type { SiteConfig } from "@/types";

export const siteConfig: SiteConfig = {
	// Used as a meta property (src/components/BaseHead.astro)
	author: "Chaikit",
	// Meta property used to construct the meta title property, found in src/components/BaseHead.astro
	title: "Chaikit Slothmdx",
	// Meta property used as the default description meta property
	description: "Astro + MDX - Blog + Podcast Starter",
	// HTML lang property, found in src/layouts/Base.astro
	lang: "en",
	// Meta property, found in src/components/BaseHead.astro
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
			url: "https://chaikit.xyz",
			name: "Chaikit"
		},
		blogTitle: "Devlog"
	},
	// src/pages file config ->
	index: {
		title: "Hello World!",
		subtitle: "Hi! This is the home page for your very own Astro + MDX blog and podcasts 🚀. Easily enter markdown and audio files to your website!",
	},
	socialLinks :{
		title: "Find me on"
	},
	about: {
		title: "About",
		subtitle: `Hi, here you can talk about yourself, your interests, and your hobbies.`,
	},
	podcast: {
		title: "Podcast",
		subtitle: "Hi! This is the podcast page for your very own Astro + MDX blog and podcasts 🚀. Easily enter audio files to your website!"
	},
	// Theme options: "Sunset Forest", "Autumn Grove", "Sunrise Orchid", "Crimson Tide"
	theme: "Crimson Tide"
};

// Social icons used in src/components/SocialList.astro
// shown on home page src/pages/index.astro
export const socialLinks: Array<{
	name: string;
	friendlyName: string;
	link: string;
}> = [
{
		name: "mdi:web",
		friendlyName: "Website",
		link: "https://www.montek.dev",
	},
	{
		name: "ic:baseline-discord",
		friendlyName: "Discord",
		link: "https://discord.com/users/702170848508903444",
	},
	{
		name: "mdi:github",
		friendlyName: "Github",
		link: "https://github.com/montekkundan",
	},
	{
		name: "mdi:twitter",
		friendlyName: "Twitter",
		link: "https://www.x.com/montekkundan/",
	},
	{
		name: "mdi:linkedin",
		friendlyName: "LinkedIn",
		link: "https://www.linkedin.com/in/montekkundan/",
	},
	{
		name: "mdi:instagram",
		friendlyName: "Instagram",
		link: "https://www.instagram.com/montekkundan/",
	},
	{
		name: "mdi:email",
		friendlyName: "email",
		link: "mailto:montekkundan@gmail.com",
	},
];


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

// The maxium number of posts and podcasts to display on a single page.
export const POSTS_PER_PAGE = 5;
export const PODCASTS_PER_PAGE = 2;
