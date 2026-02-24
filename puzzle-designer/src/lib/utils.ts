import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface Message {
	role: 'user' | 'assistant' | 'system'
	content: string
}

interface QueryOptions {
	prompt?: string
	messages?: Message[]
	stream?: boolean
	[key: string]: unknown
}

const queryURL = 'http://localhost:3000/bifrost'
export const query = async (model: string, { prompt = '', messages, stream = false, ...opts }: QueryOptions = {}) => {
	const finalMessages: Message[] =
		messages ??
		(prompt
			? [
					{
						role: 'user',
						content: prompt,
					},
			  ]
			: [])

	if (stream) {
		throw new Error('Streaming is not supported via client query; use server SSE endpoint directly.')
	}

	const res = await fetch(queryURL, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
		},
		body: JSON.stringify({
			model,
			messages: finalMessages,
			...opts,
		}),
	})

	if (!res.ok) {
		throw new Error(`Query failed with status ${res.status}`)
	}

	return res.json()
}