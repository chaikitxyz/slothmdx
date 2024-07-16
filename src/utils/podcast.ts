import type { CollectionEntry } from "astro:content";
import { getCollection } from "astro:content";

/** Note: this function filters out draft podcasts based on the environment */
export async function getAllPodcasts() {
	return await getCollection("podcast", ({ data }) => {
		return import.meta.env.PROD ? data.draft !== true : true;
	});
}

export function sortPodcastByDate(posts: Array<CollectionEntry<"podcast">>) {
	return posts.sort((a, b) => {
		const aDate = new Date(a.data.updatedDate ?? a.data.publishDate).valueOf();
		const bDate = new Date(b.data.updatedDate ?? b.data.publishDate).valueOf();
		return bDate - aDate;
	});
}


