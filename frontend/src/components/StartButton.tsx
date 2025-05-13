import {useState} from 'react';

const StartButton = () => {
    // TODO: This is currently just a counter, but should do something else later on.

    const [count, setCount] = useState<number>(0);
    const increment = () => setCount(count + 1);

    return (
    <>
        <h1> The current count is {count} </h1>
        <button onClick={increment}>
            Increment
        </button>
        <p>Click the button to increment the count.</p>
    </>
    )   
}

export default StartButton;