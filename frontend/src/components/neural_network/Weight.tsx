type WeightProperties = {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    isActive: boolean;
    activeColour?: string;
    inactiveColour?: string;
}

const Weight = ({x1, y1, x2, y2, isActive=false, activeColour="green", inactiveColour="white"}:WeightProperties) => {
    return (
        <line
            x1={x1}
            y1={y1}
            x2={x2}
            y2={y2}
            stroke={isActive ? activeColour : inactiveColour}
            strokeWidth="3"
        />
    );
}
export default Weight;