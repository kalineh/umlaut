using System.Collections;
using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;

[CustomEditor(typeof(Neural))]
public class NeuralEditor
    : Editor
{
    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();

        if (!Application.isPlaying)
            return;

        var self = target as Neural;

        if (GUILayout.Button("Reset"))
            self.Reset();
        if (GUILayout.Button("Randomize"))
            self.Randomize();
        if (GUILayout.Button("RandomizeInputs"))
            self.RandomizeInputs();
    }
}
#endif

public class Neural
    : MonoBehaviour
{
    // single neuron takes input in, and output act out
    // sigmoid(x) = 1.0f / (1.0f - exp(-x));
    // out = sigmoid(in)
    // deriv(x) = sigmoid(x) * (1.0f - sigmoid(x))

    // input to a neuron is the weighted sum of output from other neurons
    // each neuron also has a bias value; resting state, that adds to the result

    // neurons layer 0,1,2
    // 

    // data layout
    // [N0..Nn]
    // [N0W0..N0Wn]
    // ...
    // [NmW0..NmWn]

    // N0w0
    // 

    public void OnEnable()
    {
        Reset();
        Randomize();
        RandomizeInputs();
        Step();
    }

    private static float rand01() { return Random.Range(0.0f, 1.0f); }
    private static float randh() { return Random.Range(-0.5f, 0.5f); }
    private static float sigm(float x) { return x / (1.0f - Mathf.Exp(-x)); }
    private static float deriv(float x) { return sigm(x) * (1.0f - sigm(x)); }
    private static float tanh(float x) { return (float)System.Math.Tanh((float)x); }

    private int Layer0 = 4;
    private int Layer1 = 8;
    private int Layer2 = 6;

    private float[] bias0;
    private float[] bias1;
    private float[] bias2;

    private float[] state0;
    private float[] state1;
    private float[] state2;

    private float[] weights10;
    private float[] weights21;

    public void Reset()
    {
        Debug.LogFormat("Neural.Reset(): Layers: {0}, {1}, {2}", Layer0, Layer1, Layer2);

        bias0 = new float[Layer0];
        bias1 = new float[Layer1];
        bias2 = new float[Layer2];

        state0 = new float[Layer0];
        state1 = new float[Layer1];
        state2 = new float[Layer2];

        // [h0: w0..n], [h1: w0..wn]
        weights10 = new float[Layer1 * Layer0];
        weights21 = new float[Layer2 * Layer1];
    }

    public void Update()
    {
        Step();
    }

    public void Randomize()
    {
        Debug.LogFormat("Neural.Randomize()");

        // randomize biases
        for (int i = 0; i < Layer0; ++i)
            bias0[i] = randh();
        for (int i = 0; i < Layer1; ++i)
            bias1[i] = randh();
        for (int i = 0; i < Layer2; ++i)
            bias2[i] = randh();

        // randomize weights
        for (int i = 0; i < Layer1; ++i)
            for (int j = 0; j < Layer0; ++j)
                weights10[i * Layer0 + j] = randh();

        // randomize weights
        for (int i = 0; i < Layer2; ++i)
            for (int j = 0; j < Layer1; ++j)
                weights21[i * Layer1 + j] = randh();
    }

    public void RandomizeInputs()
    {
        Debug.LogFormat("Neural.RandomizeInputs()");

        // layer 0 (inputs)
        for (int i = 0; i < Layer0; ++i)
            state0[i] = (float)i / (float)(Layer0 - 1);
    }

    public void Step()
    {
        Debug.LogFormat("Neural.Step(): step...");

        // layer 1 (hidden)
        for (int i = 0; i < Layer1; ++i)
        {
            var x = bias1[i];
            for (int j = 0; j < Layer0; ++j)
                x += (state0[j] + bias0[j]) * weights10[i * Layer0 + j];
            x = tanh(x);
            state1[i] = x;
        }

        // layer 2 (output)
        for (int i = 0; i < Layer2; ++i)
        {
            var x = bias2[i];
            for (int j = 0; j < Layer1; ++j)
                x += (state1[j] + bias1[j]) * weights21[i * Layer1 + j];
            x = tanh(x);
            state2[i] = x;
        }
    }
}
