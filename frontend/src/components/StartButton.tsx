import { useState } from 'react';
import { callBackendAPI} from '../services/api';
import type { APIResponse } from '../services/api';

const StartButton = () => {

    const [info, setInfo] = useState<APIResponse | null>(null);

    const handleClick = () => {
        callBackendAPI()
            .then(setInfo)
            .catch(console.error);
    };

    return (
    <>
        <h2> Information: {info ? JSON.stringify(info) : "No info yet"} </h2>
        <button onClick={handleClick}>
            Start
        </button>
        <p>Click the button to fetch information.</p>
    </>
    )   
}

export default StartButton;