import OpenAI from "jsr:@openai/openai"

/**
 * Performs a completion using the provided schema, system prompt, and user prompt.
 * The schema instruction is automatically appended to the system prompt.
 * @param schema - The JSON schema object that the response should adhere to.
 * @param systemPrompt - The base system prompt (e.g., "You are a helpful AI assistant.").
 * @param userPrompt - The user prompt to send to the model.
 * @param model - The model to use for the completion (default: "meta-llama/Llama-3.3-70B-Instruct").
 * @returns A promise that resolves to the parsed JSON response object.
 */
export async function completeWithSchema(
    apiKey: string,
    baseURL: string,
    schema: Record<string, unknown>,
    systemPrompt: string,
    userPrompt: string,
    model: string = "meta-llama/Llama-3.3-70B-Instruct"
): Promise<[string, string | undefined]> {
    const client = new OpenAI({
        apiKey: apiKey,
        baseURL: baseURL,
    })

    const schemaStr = JSON.stringify(schema)
    const systemMessage = `${systemPrompt} Here's the json schema you need to adhere to: <schema>${schemaStr}</schema>`

    const response = await client.chat.completions.create({
        messages: [
            { role: "system", content: systemMessage },
            { role: "user", content: userPrompt },
        ],
        model: model,
        stream: false,
        response_format: {
            type: "json_schema",
            json_schema: {
                name: "response",
                schema: schema,
                strict: true,
            },
        },
    })

    const content = response.choices[0].message.content
    if (!content) {
        throw new Error("Failed to generate a response.")
    }

    let reasoning = undefined
    if ("reasoning_content" in response.choices[0].message) {
        reasoning = response.choices[0].message.reasoning_content as string
    }
    return [content, reasoning]
}
