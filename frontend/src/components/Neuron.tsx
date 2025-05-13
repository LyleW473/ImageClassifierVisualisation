type NeuronProperties = {
    x: number;
    y: number;
    radius: number;
    active: boolean;
    
}

const Neuron = ({x, y, radius, active}:NeuronProperties) => {
    return (
        <circle
            cx={x}
            cy={y}
            r={radius}
            fill={active ? "green" : "red"}
            stroke="black"
            strokeWidth="1"
        />
    );
}
export default Neuron;