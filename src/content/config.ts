import { z, defineCollection } from 'astro:content';

function removeDupsAndLowerCase(array: string[]) {
  if (!array.length) return array;
  const lowercaseItems = array.map((str) => str.toLowerCase());
  const distinctItems = new Set(lowercaseItems);
  return Array.from(distinctItems);
}

const post = defineCollection({
  type: 'content',
  schema: ({ image }) =>
    z.object({
      title: z.string().max(60),
      description: z.string().min(50).max(160),
      publishDate: z
        .string()
        .or(z.date())
        .transform((val) => new Date(val)),
      updatedDate: z
        .string()
        .optional()
        .transform((str) => (str ? new Date(str) : undefined)),
      coverImage: z
        .object({
          src: image(),
          alt: z.string(),
        })
        .optional(),
      draft: z.boolean().default(false),
      tags: z.array(z.string()).default([]).transform(removeDupsAndLowerCase),
      ogImage: z.string().optional(),
    }),
});
const podcast = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    description: z.string().min(50).max(160),
    audioUrl: z.string(),
    publishDate: z
      .string()
      .or(z.date())
      .transform((val) => new Date(val)),
    updatedDate: z
      .string()
      .optional()
      .transform((str) => (str ? new Date(str) : undefined)),
    duration: z.string(),
    size: z.number(),
    cover: z.string().optional(),
    explicit: z.boolean().default(false),
    episode: z.number(),
    season: z.number(),
    episodeType: z.enum(['full', 'trailer', 'bonus']),
    draft: z.boolean().default(false),
    ogImage: z.string().optional(),
  }),
});

export const collections = { post, podcast };
