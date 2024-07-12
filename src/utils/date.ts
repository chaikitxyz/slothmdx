import { siteConfig } from "@/site-config";

const dateFormat = new Intl.DateTimeFormat(siteConfig.date.locale, siteConfig.date.options);

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
