//
//  OnboardingView.swift
//

import SwiftUI

struct OnboardingView: View {

    @AppStorage("hasSeenOnboarding") private var hasSeenOnboarding: Bool = false

    var body: some View {
        VStack(spacing: 24) {

            // ⭐️ Move your intro text from ContentView to here
            Text("Welcome to Pouch Cell Inspector")
                .font(.largeTitle)
                .fontWeight(.bold)
                .multilineTextAlignment(.center)
            Text("Use the camera to identify if your lithium ion battery is normal.")
            Text("Simply take a picture and it will give you information about your lithium ion battery. Tap continue to get started.")

                            .font(.body)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            Button(action: {
                hasSeenOnboarding = true
            }) {
                Text("Continue")
                    .font(.headline)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(12)
            }
            .padding(.horizontal)
        }
        .padding()
    }
}

