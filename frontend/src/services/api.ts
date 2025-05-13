import axios from 'axios';

export interface PredictionResponse {
    status: string;
    message?: string;
    confidence: number;
    predicted_class_name: string;
    actual_class_name: string
}

export const callBackendAPI = async () => {
    const response = await axios.get<PredictionResponse>("http://localhost:8000/predict")
    if (response.status !== 200) {
        throw new Error(`Error: ${response.status}`);
    }
    return response.data;
}