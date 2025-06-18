
import Section from './Section';
import type { PredictionResponse } from '../services/api';

type ResultBoxProps = {
    // originalImagePath?: string; // Path to the original image in the public folder
    // predictedClassName: string; // The class name predicted by the model
    // actualClassName: string; // The actual class name of the image
    // confidence: number; // The confidence level of the prediction, between 0 and 1
    result: PredictionResponse | null
}

const ResultBox = ({ result }: ResultBoxProps) => {

    let predictedClassName: string;
    let actualClassName: string;
    let confidence: number;
    let originalImagePath: string;

    if (!result) {
        predictedClassName = "N/A";
        actualClassName = "N/A";
        confidence = 0;
        originalImagePath = "placeholder_image.jpg"; // Placeholder image path
    }
    else {
        predictedClassName = result.predictedClassName;
        actualClassName = result.actualClassName;
        confidence = result.confidence;
        originalImagePath = result.originalImagePath
    }
    console.log(originalImagePath);

    return (
        <div style={{ display: 'flex', gap: '20px', justifyContent: 'center', alignItems: 'flex-start'}}>
            <Section title={"Original Image"}>
            <img src={originalImagePath} alt="Original" style={{ maxWidth: '100%' }} />
            </Section>

            <Section title={"Metadata"}>
                <p>Predicted class: {predictedClassName}</p>
                <p>Actual class: {actualClassName}</p>
                <p>Confidence: {confidence * 100}%</p>
            </Section>
        </div>
    );
}
export default ResultBox;