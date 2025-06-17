
import { callBackendAPI} from '../services/api';
import type { PredictionResponse } from '../services/api';

type StartButtonProperties = {
    onFetch: (info: PredictionResponse) => void;
}

const StartButton = ({ onFetch }: StartButtonProperties) => {
    const handleClick = () => {
        callBackendAPI()
            // Add save and add to the predictionREsponse (attach the image path)
            .then(onFetch)
            .catch(console.error);
    };

    return (
    <>
        <button onClick={handleClick}>
            Start
        </button>
        <p>Click the button to fetch information.</p>
    </>
    )   
}

export default StartButton;