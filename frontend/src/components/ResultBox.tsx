
import Section from './Section';

type ResultBoxProps = {
    originalImagePath?: string; // Path to the original image in the public folder
    predictedClassName: string; // The class name predicted by the model
    actualClassName: string; // The actual class name of the image
    confidence: number; // The confidence level of the prediction, between 0 and 1
}

const ResultBox = ({ originalImagePath, predictedClassName, actualClassName, confidence }: ResultBoxProps) => {
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