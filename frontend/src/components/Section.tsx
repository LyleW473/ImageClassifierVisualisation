import type { ReactNode } from "react";

type SectionProperties = {
    title?: string;
    children: ReactNode;
}

const Section = ({title="Example Section", children}: SectionProperties) => {
    return (
        <section>
            <h2>{title}</h2>
            <p>{children}</p>
        </section>
    )
}
export default Section;