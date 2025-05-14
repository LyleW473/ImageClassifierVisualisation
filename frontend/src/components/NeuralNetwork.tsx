import NeuronLayer from "./NeuronLayer";


type NeuralNetworkProperties = {
    topLeftNeuronX: number;
    topLeftNeuronY: number;
    numLayers: number;
    neuronsPerLayer: number[];
    neuronXGap: number;
    neuronRadius: number;
    neuronPaddingX?: number;
    neuronPaddingY?: number;
    neuronSpacingY: number;
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
    neuronPaddingX=0,
    neuronPaddingY=0,
    neuronSpacingY,
    neuronActiveColour,
    neuronInactiveColour
    }: NeuralNetworkProperties) => {
    
    const layers = [];
    for (let i = 0; i < numLayers; i++) {
        const layerXStart = topLeftNeuronX + (i * (neuronRadius * 2 + neuronXGap));
        const neuralLayer = <NeuronLayer
            key={i}
            rectX={layerXStart}
            rectY={topLeftNeuronY}
            paddingX={neuronPaddingX}
            paddingY={neuronPaddingY}
            numNeurons={neuronsPerLayer[i]}
            neuronRadius={neuronRadius}
            neuronSpacingY={neuronSpacingY}
            neuronActiveColour={neuronActiveColour}
            neuronInactiveColour={neuronInactiveColour}
        />  
        layers.push(neuralLayer);
    }
    return <>{layers}</>
}

export default NeuralNetwork;