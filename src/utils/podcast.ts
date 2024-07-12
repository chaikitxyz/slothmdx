import type { CollectionEntry } from "astro:content";
import { getCollection } from "astro:content";

/** Note: this function filters out draft podcasts based on the environment */
export async function getAllPodcasts() {
	return await getCollection("podcast", ({ data }) => {
		return import.meta.env.PROD ? data.draft !== true : true;
	});
}

export function sortPodcastByDate(entries: CollectionEntry<"podcast">[]): CollectionEntry<"podcast">[] {
	return entries.sort((a, b) => {
	  const dateA = new Date(a.data.publishDate).getTime();
	  const dateB = new Date(b.data.publishDate).getTime();
	  return dateB - dateA;
	});
  }

