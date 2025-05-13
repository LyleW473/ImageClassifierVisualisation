export type APIResponse = {
    status: string;
    message?: string;
}

export async function callBackendAPI(): Promise<APIResponse> {
    const response = await fetch("http://localhost:8000/predict")
    if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
    }
    return await response.json();
}