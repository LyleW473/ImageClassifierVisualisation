import Neuron from './Neuron';

type NeuronLayerProperties = {
    rectX: number;
    rectY: number;
    numNeurons: number;
    neuronRadius: number;
    paddingX: number;
    paddingY: number;
    neuronSpacingY: number;
    neuronActiveColour?: string;
    neuronInactiveColour?: string;
}

const NeuronLayer = (
        {
            rectX,
            rectY,
            numNeurons, 
            neuronRadius, 
            paddingX,
            paddingY,
            neuronSpacingY, 
            neuronActiveColour, 
            neuronInactiveColour
        }: NeuronLayerProperties) => {

    const neurons = [];
    
    // Calculate the height/width of the layer
    const totalNeuronHeight = numNeurons * (2 * neuronRadius);
    const totalSpacing = (numNeurons - 1) * neuronSpacingY;
    const layerHeight = totalNeuronHeight + totalSpacing + 2 * paddingY;
    const layerWidth = 2 * neuronRadius + 2 * paddingX;

    // Calculate the offset for the first neuron
    const offsetY = neuronRadius + paddingY;
    
    for (let i = 0; i < numNeurons; i++) {
        const cx = rectX + layerWidth / 2;
        const cy = offsetY + rectY + (i * ((neuronRadius * 2) + neuronSpacingY));
        
        neurons.push(
            <Neuron
                key={i}
                cx={cx}
                cy={cy}
                radius={neuronRadius}
                active={false}
                activeColour={neuronActiveColour}
                inactiveColour={neuronInactiveColour}
            />
        )
    }
    
    // Return array of neurons (essentially <circle> elements)
    return (
        <>
            <rect
                x={rectX}
                y={rectY}
                width={layerWidth}
                height={layerHeight}
                fill="#504343"
                stroke="black"
                strokeWidth={3}
            />
            {neurons}
        </>
    )
}
export default NeuronLayer;