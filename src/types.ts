export type SiteConfig = {
	author: string;
	title: string;
	description: string;
	lang: string;
	ogLocale: string;
	date: {
		locale: string | string[] | undefined;
		options: Intl.DateTimeFormatOptions;
	};
	header: {
		logo: {
			url: string;
			name: string;
		};
		blogTitle: string;
	};
	index: {
		title: string;
		subtitle: string;
	};
	about: {
		title: string;
		subtitle: string;
	};
	theme: string;
};

export type PaginationLink = {
	url: string;
	text?: string;
	srLabel?: string;
};

export type SiteMeta = {
	title: string;
	description?: string;
	ogImage?: string | undefined;
	articleDate?: string | undefined;
};

export type Author = {
	type: string;
	name: string;
	photo: string;
	url: string;
};

export type Content = {
	"content-type": string;
	value: string;
	html: string;
	text: string;
};

export type Rels = {
	canonical: string;
};

export type Summary = {
	"content-type": string;
	value: string;
};