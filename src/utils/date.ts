import siteConfig from "src/siteConfig.json";

const dateFormat = new Intl.DateTimeFormat(siteConfig.date.locale, {
	day: siteConfig.date.options.day as "numeric" | "2-digit",
	month: siteConfig.date.options.month as "numeric" | "2-digit" | "narrow" | "short" | "long",
	year: siteConfig.date.options.year as "numeric" | "2-digit"
  });

export function getFormattedDate(
	date: string | number | Date,
	options?: Intl.DateTimeFormatOptions
) {
	const parsedDate = new Date(date);
	if (isNaN(parsedDate.getTime())) {
		throw new Error("Invalid date provided to getFormattedDate function.");
	}

	if (typeof options !== "undefined") {
		return parsedDate.toLocaleDateString(siteConfig.date.locale, {
			...(siteConfig.date.options as Intl.DateTimeFormatOptions),
			...options,
		});
	}

	return dateFormat.format(parsedDate);
}
