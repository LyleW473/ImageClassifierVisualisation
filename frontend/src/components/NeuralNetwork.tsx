import NeuronLayer from "./NeuronLayer";


type NeuralNetworkProperties = {
    topLeftNeuronX: number;
    topLeftNeuronY: number;
    numLayers: number;
    neuronsPerLayer: number[];
    neuronXGap: number;
    neuronRadius: number;
    neuronSpacing: number;
    neuronActiveColour?: string;
    neuronInactiveColour?: string;
}

const NeuralNetwork = ({
    topLeftNeuronX,
    topLeftNeuronY,
    numLayers,
    neuronsPerLayer,
    neuronXGap,
    neuronRadius,
    neuronSpacing,
    neuronActiveColour,
    neuronInactiveColour
    }: NeuralNetworkProperties) => {
    
    const layers = [];
    for (let i = 0; i < numLayers; i++) {
        const layerXStart = topLeftNeuronX + (i * (neuronRadius * 2 + neuronXGap));
        const neuralLayer = <NeuronLayer
            key={i}
            topLeftNeuronX={layerXStart}
            topLeftNeuronY={topLeftNeuronY}
            numNeurons={neuronsPerLayer[i]}
            neuronRadius={neuronRadius}
            neuronSpacing={neuronSpacing}
            neuronActiveColour={neuronActiveColour}
            neuronInactiveColour={neuronInactiveColour}
        />  
        layers.push(neuralLayer);
    }
    return <>{layers}</>
}

export default NeuralNetwork;