type NeuronProperties = {
    cx: number;
    cy: number;
    radius: number;
    active: boolean;
    activeColour?: string;
    inactiveColour?: string;
}

const Neuron = ({cx, cy, radius, active, activeColour="green", inactiveColour="red"}:NeuronProperties) => {
    return (
        <circle
            cx={cx}
            cy={cy}
            r={radius}
            fill={active ? activeColour : inactiveColour}
            stroke="black"
            strokeWidth="1"
        />
    );
}
export default Neuron;