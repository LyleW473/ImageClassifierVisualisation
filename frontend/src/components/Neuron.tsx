type NeuronProperties = {
    x: number;
    y: number;
    radius: number;
    active: boolean;
    activeColour?: string;
    inactiveColour?: string;
}

const Neuron = ({x, y, radius, active, activeColour="green", inactiveColour="red"}:NeuronProperties) => {
    return (
        <circle
            cx={x}
            cy={y}
            r={radius}
            fill={active ? activeColour : inactiveColour}
            stroke="black"
            strokeWidth="1"
        />
    );
}
export default Neuron;