import axios from 'axios';

export interface PredictionResponse {
    status: string;
    confidence: number;
    predictedClassName: string;
    actualClassName: string
    originalImagePath: string; // Path to the original image in the public folder
}

export const callBackendAPI = async ():Promise<PredictionResponse> => {
    // TODO: Call the backend API instead
    
    // Call ML backend for prediction
    const response = await axios.get<PredictionResponse>('/predict')
    if (response.status !== 200) {
        throw new Error(`Error: ${response.status}`);
    }
    return response.data;
}