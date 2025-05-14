import Neuron from './Neuron';

type NeuronLayerProperties = {
    topLeftNeuronX: number;
    topLeftNeuronY: number;
    numNeurons: number;
    neuronRadius: number;
    neuronSpacing: number;
    neuronActiveColour: string;
    neuronInactiveColour: string;
}

const NeuronLayer = (
        {
            topLeftNeuronX, 
            topLeftNeuronY, 
            numNeurons, 
            neuronRadius, 
            neuronSpacing, 
            neuronActiveColour, 
            neuronInactiveColour
        }: NeuronLayerProperties) => {
    
    const neurons = [];
    for (let i = 0; i < numNeurons; i++) {
        const x = topLeftNeuronX
        const y = topLeftNeuronY + (i * (neuronRadius * 2 + neuronSpacing));

        neurons.push(
            <Neuron
                key={i}
                x={x}
                y={y}
                radius={neuronRadius}
                active={false}
                activeColour={neuronActiveColour}
                inactiveColour={neuronInactiveColour}
            />
        )
    }
    // Return array of neurons (essentially <circle> elements)
    return <>{neurons}</>
}
export default NeuronLayer;