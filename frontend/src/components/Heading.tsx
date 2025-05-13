type HeadingProperties = {
    title: string;
}

const Heading = ({title}: HeadingProperties) => {
    return <h1>{title}</h1>
}
export default Heading;