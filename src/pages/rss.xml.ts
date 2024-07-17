import rss from "@astrojs/rss";
import siteConfig from "src/siteConfig.json";
import { getAllPosts } from "@/utils";

export const GET = async () => {
	const posts = await getAllPosts();

	return rss({
		title: siteConfig.title,
		description: siteConfig.description,
		site: import.meta.env.SITE,
		items: posts.map((post: { data: { title: any; description: any; publishDate: any; }; slug: any; }) => ({
			title: post.data.title,
			description: post.data.description,
			pubDate: post.data.publishDate,
			link: `posts/${post.slug}`,
		})),
	});
};
