import Heading from './components/Heading'
import Section from './components/Section'
import StartButton from './components/StartButton'
import Neuron from './components/Neuron'

function App() {
  return (
    <>
      <Heading title={"Hello World"} />
      <Section title={"Example Section"}>
          This is an example section
      </Section>
      <svg width="1000" height="1000">
        <Neuron x={100} y={100} radius={100} active={false}></Neuron>
        <Neuron x={100} y={400} radius={100} active={false}></Neuron>
        <Neuron x={100} y={700} radius={100} active={false}></Neuron>
      </svg>
      <StartButton />
    </>
  )
}

export default App
