import NeuronLayer from "./NeuronLayer";


type NeuralNetworkProperties = {
    canvasWidth: number;
    canvasHeight: number;
    neuronsPerLayer: number[];
    gapBetweenLayersX: number;
    neuronRadius: number;
    neuronPaddingX?: number;
    neuronPaddingY?: number;
    neuronSpacingY: number;
    neuronActiveColour?: string;
    neuronInactiveColour?: string;
}

const NeuralNetwork = ({
    canvasWidth,
    canvasHeight,
    neuronsPerLayer,
    gapBetweenLayersX,
    neuronRadius,
    neuronPaddingX=0,
    neuronPaddingY=0,
    neuronSpacingY,
    neuronActiveColour,
    neuronInactiveColour
    }: NeuralNetworkProperties) => {
    
    const layers = [];
    const numLayers = neuronsPerLayer.length;
    const layerWidth = 2 * neuronRadius + 2 * neuronPaddingX;
    const totalNetworkWidth = numLayers * layerWidth + (numLayers - 1) * gapBetweenLayersX;
    const networkStart = (canvasWidth - totalNetworkWidth) / 2;

    for (let i = 0; i < numLayers; i++) {
        const numNeurons = neuronsPerLayer[i];

        // Center the layer vertically
        const layerHeight = numNeurons * (2 * neuronRadius) + (numNeurons - 1) * neuronSpacingY + 2 * neuronPaddingY;
        const layerYStart = (canvasHeight / 2) - (layerHeight / 2); 
        
        // Calculate the x position of the layer (ensuring that the layers are spaced out evenly throughout the canvas)
        const layerXStart = networkStart + (i * (layerWidth + gapBetweenLayersX));

        const neuralLayer = <NeuronLayer
            key={i}
            rectX={layerXStart}
            rectY={layerYStart}
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